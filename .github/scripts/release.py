#!/usr/bin/env python3
import os, shutil
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from argparse import ArgumentParser

# For interfacing Git REST API
import re
import datetime
from github import Github
from github.GithubException import *
from github.InputGitAuthor import InputGitAuthor

# Local interfacing of repo
from git import Repo
from git import PushInfo

import zipfile

from versioning import update_version_number_in_freertos_component
from versioning import update_freertos_version_macros

from packager import prune_result_tree, prune_result_tree_v2
from packager import RELATIVE_FILE_EXCLUDES as FREERTOS_RELATIVE_FILE_EXCLUDES

# PyGithub Git -  https://github.com/PyGithub/PyGithub
# PyGithub Docs - https://pygithub.readthedocs.io/en/latest/github_objects
# REST API used by PyGithub - https://developer.github.com/v3/

indent_level = 0

# Files/Directories not to be included in the FreeRTOS-Kernel release.
KERNEL_RELEASE_EXCLUDE_FILES = [
    '.git',
    '.github',
    '.gitignore',
    '.gitmodules'
]

def logIndentPush():
    global indent_level
    indent_level += 4

def logIndentPop():
    global indent_level
    indent_level -= 4

    indent_level = max(indent_level, 0)

def info(msg, end='\n'):
    print(f"[INFO]: {' ' * indent_level}{str(msg)}", end=end, flush=True)

def warning(msg):
    print(f"[WARNING]: {' ' * indent_level}{str(msg)}", flush=True)

def error(msg):
    print(f"[ERROR]: {' ' * indent_level}{str(msg)}", flush=True)

def debug(msg):
    print(f"[DEBUG]: {' ' * indent_level}{str(msg)}", flush=True)

# Callback for progress updates. For long spanning gitpython commands
def printDot(op_code, cur_count, max_count=None, message=''):
    if max_count is None or cur_count == max_count:
        print('.', end='')

class BaseRelease:
    def __init__(self, mGit, version, commit='HEAD', git_ssh=False, git_org='FreeRTOS', repo_path=None, branch='main', do_not_push=False):
        self.version = version
        self.tag_msg = 'Autocreated by FreeRTOS Git Tools.'
        self.commit = commit
        self.git_ssh = git_ssh
        self.git_org = git_org
        self.repo_path = repo_path
        self.local_repo = None
        self.branch = branch
        self.commit_msg_prefix = '[AUTO][RELEASE]: '
        self.description = ''
        self.mGit = mGit # Save a handle to the authed git session
        self.do_not_push = do_not_push

        if self.repo_path:
            info(f'Sourcing "{self.repo_path}" to make local commits')
            self.local_repo = Repo(self.repo_path)

    def CheckRelease(self):
        '''
        Sanity checks for the release. Verify version number format. Check zip size.
        Ensure version numbers were updated, etc.
        '''
        assert False, 'Add release check'

    def hasTag(self, tag):
        remote_tags = self.repo.get_tags()
        return any(t.name == tag for t in remote_tags)

    def commitChanges(self, msg):
        assert self.local_repo != None, 'Failed to commit. Git repo uninitialized.'

        info(f'Committing: "{msg}"')
        self.local_repo.git.add(update=True)
        commit = self.local_repo.index.commit(msg)

    def getRemoteEndpoint(self, repo_name):
        if self.git_ssh:
            return f'git@github.com:{repo_name}.git'
        else:
            return f'https://github.com/{repo_name}.git'

    def printReleases(self):
        releases = self.repo.get_releases()
        for r in releases:
            print(r)

    def pushLocalCommits(self, force=False):
        if self.do_not_push:
            info('Skipping to push local commits...')
        else:
            info('Pushing local commits...')
            push_infos = self.local_repo.remote('origin').push(force=force)

            # Check for any errors
            for push_info in push_infos:
                assert (
                    0 == push_info.flags & PushInfo.ERROR
                ), f'Failed to push changes to {str(push_info)}'

    def pushTag(self):
        if self.do_not_push:
            info(f'Skipping to push tag "{self.tag}"')
        else:
            # Overwrite existing tags
            info(f'Pushing tag "{self.tag}"')
            tag_info = self.local_repo.create_tag(self.tag, message=self.tag_msg, force=True)
            self.local_repo.git.push(tags=True, force=True)

    def deleteTag(self):
        # Remove from remote
        if self.tag in self.local_repo.tags:
            info(f'Deleting tag "{self.tag}"')
            self.local_repo.remote('origin').push(f':{self.tag}')
        else:
            info(f'A tag does not exists for "{self.tag}". No need to delete.')

    def updateSubmodulePointer(self, rel_path, ref):
        submodule = Repo(rel_path)
        submodule.remote('origin').fetch()
        submodule.git.checkout(ref)

    def updateFileHeaderVersions(self, old_version_substrings, new_version_string):
        info(f'Updating file header versions for "{self.version}"...', end='')
        n_updated = 0
        n_updated += update_version_number_in_freertos_component(self.repo_path,
                                                                 '.',
                                                                 old_version_substrings,
                                                                 new_version_string,
                                                                 exclude_hidden=True)

        n_updated += update_version_number_in_freertos_component(os.path.join('.github', 'scripts'),
                                                                 self.repo_path,
                                                                 old_version_substrings,
                                                                 new_version_string,
                                                                 exclude_hidden=False)

        print('...%d Files updated.' % n_updated)

        self.commitChanges(
            f'{self.commit_msg_prefix}Bump file header version to "{self.version}"'
        )

    def deleteGitRelease(self):
        info(f'Deleting git release endpoint for "{self.tag}"')

        try:
            self.repo.get_release(self.tag).delete_release()
        except UnknownObjectException:
            info(f'A release endpoint does not exist for "{self.tag}". No need to erase.')
        except:
            assert False, 'Encountered error while trying to delete git release endpoint'

    def rollbackAutoCommits(self, n_autocommits=2, n_search=25):
        info(f'Rolling back "{self.tag}" autocommits')

        if self.tag not in self.local_repo.tags:
            error(f'Could not find a SHA to rollback to for tag "{self.tag}"')
            return False

        # Search for auto release SHAs that match the release tag SHA
        tag_commit = self.local_repo.tag(f'refs/tags/{self.tag}').commit
        prior_commit = self.local_repo.commit(tag_commit.hexsha + '~%d' % n_autocommits)
        for n_commits_searched, commit in enumerate(self.local_repo.iter_commits()):
            if n_commits_searched > n_search:
                error('Exhaustively searched but could not find tag commit to rollback')
                return False

            if (self.commit_msg_prefix in commit.message
                    and commit.hexsha == tag_commit.hexsha
                    and self.version in commit.message):

                info(
                    f'Found matching tag commit {tag_commit.hexsha}. Reverting to prior commit {prior_commit.hexsha}'
                )

                # Found the commit prior to this autorelease. Revert back to it then push
                self.local_repo.git.reset(prior_commit.hexsha, hard=True)
                self.pushLocalCommits(force=True)
                return True

        return False

    def restorePriorToRelease(self):
        info(f'Restoring "main" to just before autorelease:{self.version}')

        self.deleteGitRelease()
        self.rollbackAutoCommits()
        self.deleteTag()
        self.pushLocalCommits(force=True)


class KernelRelease(BaseRelease):
    def __init__(self, mGit, version, commit='HEAD', git_ssh=False, git_org='FreeRTOS', repo_path=None, branch='main', main_br_version='', do_not_push=False):
        super().__init__(mGit, version, commit=commit, git_ssh=git_ssh, git_org=git_org, repo_path=repo_path, branch=branch, do_not_push=do_not_push)

        self.repo_name = f'{self.git_org}/FreeRTOS-Kernel'
        self.repo = mGit.get_repo(self.repo_name)
        self.tag = f'V{version}'
        self.description = 'Contains source code for the FreeRTOS Kernel.'
        self.zip_path = f'FreeRTOS-KernelV{self.version}.zip'
        self.main_br_version = main_br_version

        # Parent ctor configures local_repo if caller chooses to source local repo from repo_path.
        if self.repo_path is None:
            self.repo_path = 'tmp-release-freertos-kernel'
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)

            # Clone the target repo for creating the release autocommits
            remote_name = self.getRemoteEndpoint(self.repo_name)
            info(f'Downloading {remote_name}@{commit} to baseline auto-commits...', end='')
            self.local_repo = Repo.clone_from(remote_name, self.repo_path, progress=printDot, branch=self.branch)

        # In case user gave non-HEAD commit to baseline
        self.local_repo.git.checkout(commit)

        print()

    def updateVersionMacros(self, version_str):
        info(f'Updating version macros in task.h for "{version_str}"')

        # Extract major / minor / build from the version string.
        ver = re.search(r'([\d.]+)', version_str)[1]
        (major, minor, build) = ver.split('.')
        update_freertos_version_macros(os.path.join(self.repo_path, 'include', 'task.h'), version_str, major, minor, build)

        self.commitChanges(
            f'{self.commit_msg_prefix}Bump task.h version macros to "{version_str}"'
        )

    def createReleaseZip(self):
        '''
        At the moment, the only asset we upload is the source code.
        '''
        zip_name = f'FreeRTOS-KernelV{self.version}'
        info(f'Packaging "{zip_name}"')
        logIndentPush()

        # This path name is retained in zip, so we don't name it 'tmp-*' but
        # rather keep it consistent.
        rel_repo_path = zip_name

        # Clean up any old work from previous runs.
        if os.path.exists(rel_repo_path):
            shutil.rmtree(rel_repo_path)

        # Download a fresh copy for packaging.
        info(f'Downloading fresh copy of {zip_name} for packing...', end='')
        packaged_repo = Repo.clone_from(
            self.getRemoteEndpoint(self.repo_name),
            rel_repo_path,
            multi_options=['--depth=1', f'-b{self.tag}', '--recurse-submodules'],
            progress=printDot,
            branch=self.branch,
        )
        print()

        # Prune then zip package.
        info('Pruning from release zip...', end='')
        files_pruned = prune_result_tree_v2(rel_repo_path, KERNEL_RELEASE_EXCLUDE_FILES)
        print('...%d Files Removed.' % len(files_pruned))

        info(f'Compressing "{self.zip_path}"...')
        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip:
            for root, dirs, files in os.walk(rel_repo_path):
                for file in files:
                    # For some strange reason, we have broken symlinks...avoid these.
                    file_path = os.path.join(root, file)
                    if os.path.islink(file_path) and not os.path.exists(file_path):
                        warning(f'Skipping over broken symlink "{file_path}"')
                    else:
                        zip.write(file_path)

        logIndentPop()

    def createGitRelease(self):
        '''
        Creates/Overwrites release identified by target tag
        '''
        if self.do_not_push:
            info(f'Skipping creating git release endpoint for "{self.tag}"...')
        else:
            # If this release already exists, delete it
            try:
                release_queried = self.repo.get_release(self.tag)

                info(f'Overwriting existing git release endpoint for "{self.tag}"...')
                release_queried.delete_release()
            except UnknownObjectException:
                info(f'Creating git release endpoint for "{self.tag}"...')

            # Create the release asset to upload.
            self.createReleaseZip()

            # Create the new release endpoint at upload assets
            release = self.repo.create_git_release(
                tag=self.tag,
                name=f'V{self.version}',
                message=self.description,
                draft=False,
                prerelease=False,
            )
            info('Uploading release asssets...')
            release.upload_asset(
                self.zip_path,
                name=f'FreeRTOS-KernelV{self.version}.zip',
                content_type='application/zip',
            )

    def autoRelease(self):
        info(f'Auto-releasing FreeRTOS Kernel V{self.version}')

        # Determine if we need to set a separate version macros for the main branch
        if (self.commit == 'HEAD') and len(self.main_br_version) > 0 and (self.main_br_version != self.version):
            # Update version macros for main branch
            self.updateVersionMacros(self.main_br_version)

            # Push the branch
            self.pushLocalCommits()

            # Revert the last commit in our working git repo
            self.local_repo.git.reset('--hard','HEAD^')

        # Update the version macros
        self.updateVersionMacros(self.version)

        if (self.commit == 'HEAD') and (self.main_br_version == self.version):
            # Share a task.h version number commit for main branch and release tag)
            self.pushLocalCommits()

        # When baselining off a non-HEAD commit, main is left unchanged by tagging a detached HEAD,
        # applying the autocommits, tagging, and pushing the new tag data to remote.
        # However in the detached HEAD state we don't have a branch to push to, so we skip

        # Update the header in each c/assembly file
        self.updateFileHeaderVersions(
            ['FreeRTOS Kernel V', 'FreeRTOS Kernel <DEVELOPMENT BRANCH>'],
            f'FreeRTOS Kernel V{self.version}',
        )

        self.pushTag()

        self.createGitRelease()

        info('Kernel release done.')



class FreertosRelease(BaseRelease):
    def __init__(self, mGit, version, commit, git_ssh=False, git_org='FreeRTOS', repo_path=None, branch='main', do_not_push=False):
        super().__init__(mGit, version, commit, git_ssh=git_ssh, git_org=git_org, repo_path=repo_path, branch=branch, do_not_push=do_not_push)

        self.repo_name = f'{self.git_org}/FreeRTOS'
        self.repo = mGit.get_repo(self.repo_name)
        self.tag = self.version
        self.description = 'Contains source code and example projects for the FreeRTOS Kernel and FreeRTOS+ libraries.'
        self.zip_path = f'FreeRTOSv{self.version}.zip'

        # Download a fresh copy of local repo for making autocommits, if necessary
        if self.repo_path is None:
            self.repo_path = 'tmp-release-freertos'

            # Clean up any old work from previous runs
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)

            # Clone the target repo for creating the release autocommits
            remote_name = self.getRemoteEndpoint(self.repo_name)
            info(f'Downloading {remote_name}@{commit} to baseline auto-commits...', end='')
            self.local_repo = Repo.clone_from(remote_name, self.repo_path, progress=printDot, branch=self.branch)

        # In support of non-HEAD baselines
        self.local_repo.git.checkout(commit)
        print()

    def isValidManifestYML(self, path_yml):
        assert False, 'Unimplemented'

    def updateSubmodulePointers(self):
        '''
        Reads the 'manifest.yml' file from the local FreeRTOS clone that is being used to stage the commits
        '''

        info('Initializing first level of submodules...')
        self.local_repo.submodule_update(init=True, recursive=False)

        # Read YML file
        path_manifest = os.path.join(self.repo_path, 'manifest.yml')
        assert os.path.exists(path_manifest), 'Missing manifest.yml'
        with open(path_manifest, 'r') as fp:
            manifest_data = fp.read()
        yml = load(manifest_data, Loader=Loader)
        assert 'dependencies' in yml, 'Manifest YML parsing error'

        # Update all the submodules per yml
        logIndentPush()
        for dep in yml['dependencies']:
            assert 'version' in dep, 'Failed to parse submodule tag from manifest'
            assert 'repository' in dep and 'path' in dep['repository'], 'Failed to parse submodule path from manifest'
            submodule_path = dep['repository']['path']
            submodule_tag  = dep['version']

            # Update the submodule to point to version noted in manifest file
            info('%-20s : %s' % (dep['name'], submodule_tag))
            self.updateSubmodulePointer(os.path.join(self.repo_path, submodule_path), submodule_tag)
        logIndentPop()

        self.commitChanges(
            f'{self.commit_msg_prefix}Bump submodules per manifest.yml for V{self.version}'
        )

    def createReleaseZip(self):
        '''
        At the moment, the only asset we upload is the
        '''
        zip_name = f'FreeRTOSv{self.version}'
        info(f'Packaging "{zip_name}"')
        logIndentPush()

        # This path name is retained in zip, so we don't name it 'tmp-*' but rather keep it consistent with previous
        # packaging
        rel_repo_path = zip_name

        # Clean up any old work from previous runs
        if os.path.exists(rel_repo_path):
            shutil.rmtree(rel_repo_path)

        # Download a fresh copy for packaging
        info(f'Downloading fresh copy of {zip_name} for packing...', end='')
        packaged_repo = Repo.clone_from(
            self.getRemoteEndpoint(self.repo_name),
            rel_repo_path,
            multi_options=['--depth=1', f'-b{self.tag}', '--recurse-submodules'],
            progress=printDot,
            branch=self.branch,
        )
        print()

        # Prune then zip package
        info('Pruning from release zip...', end='')
        files_pruned = prune_result_tree(rel_repo_path, FREERTOS_RELATIVE_FILE_EXCLUDES)
        print('...%d Files Removed.' % len(files_pruned))

        info(f'Compressing "{self.zip_path}"...')
        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip:
            for root, dirs, files in os.walk(rel_repo_path):
                for file in files:
                    # For some strange reason, we have broken symlinks...avoid these
                    file_path = os.path.join(root, file)
                    if os.path.islink(file_path) and not os.path.exists(file_path):
                        warning(f'Skipping over broken symlink "{file_path}"')
                    else:
                        zip.write(file_path)

        logIndentPop()

    def createGitRelease(self):
        '''
        Creates/Overwrites release identified by target tag
        '''
        if self.do_not_push:
            info(f'Skipping creating git release endpoint for "{self.tag}"...')
        else:
            # If this release already exists, delete it
            try:
                release_queried = self.repo.get_release(self.tag)

                info(f'Overwriting existing git release endpoint for "{self.tag}"...')
                release_queried.delete_release()
            except UnknownObjectException:
                info(f'Creating git release endpoint for "{self.tag}"...')

            # Create the new release endpoind at upload assets
            release = self.repo.create_git_release(
                tag=self.tag,
                name=f'FreeRTOSv{self.version}',
                message=self.description,
                draft=False,
                prerelease=False,
            )

            info('Uploading release asssets...')
            release.upload_asset(
                self.zip_path,
                name=f'FreeRTOSv{self.version}.zip',
                content_type='application/zip',
            )

    def autoRelease(self):
        info(f'Auto-releasing FreeRTOS V{self.version}')

        self.updateFileHeaderVersions(
            [
                'FreeRTOS Kernel V',
                'FreeRTOS V',
                'FreeRTOS Kernel <DEVELOPMENT BRANCH>',
                'FreeRTOS <DEVELOPMENT BRANCH>',
            ],
            f'FreeRTOS V{self.version}',
        )
        self.updateSubmodulePointers()
        # When baselining off a non-HEAD commit, main is left unchanged by tagging a detached HEAD,
        # applying the autocommits, tagging, and pushing the new tag data to remote.
        # However in the detached HEAD state we don't have a branch to push to, so we skip
        if self.commit == 'HEAD':
            self.pushLocalCommits()

        self.pushTag()
        self.createReleaseZip()
        self.createGitRelease()

        info('Core release done.')

def configure_argparser():
    parser = ArgumentParser(description='FreeRTOS Release tool')

    parser.add_argument('git_org',
                        type=str,
                        metavar='GITHUB_ORG',
                        help='Git organization owner for FreeRTOS and FreeRTOS-Kernel. (i.e. "<git-org>/FreeRTOS.git")')

    parser.add_argument('--new-core-version',
                        default=None,
                        required=False,
                        help='FreeRTOS Standard Distribution Version to replace old version. (Ex. "FreeRTOS V202012.00")')

    parser.add_argument('--core-commit',
                        default='HEAD',
                        required=False,
                        metavar='GITHUB_SHA',
                        help='Github SHA to baseline autorelease')

    parser.add_argument('--rollback-core-version',
                        default=None,
                        required=False,
                        help='Reset "main" to state prior to autorelease of given core version')

    parser.add_argument('--core-repo-path',
                        type=str,
                        default=None,
                        required=False,
                        help='Instead of downloading from git, use existing local repos for autocommits')

    parser.add_argument('--core-repo-branch',
                        type=str,
                        default='main',
                        required=False,
                        help='Branch of FreeRTOS hub repository to release.')

    parser.add_argument('--new-kernel-version',
                        default=None,
                        required=False,
                        help='Reset "main" to just before the autorelease for the specified kernel version")')

    parser.add_argument('--new-kernel-main-br-version',
                        default='',
                        required=False,
                        help='Set the version in task.h on the kernel main branch to the specified value.')

    parser.add_argument('--kernel-commit',
                        default='HEAD',
                        required=False,
                        metavar='GITHUB_SHA',
                        help='Github SHA to baseline autorelease')

    parser.add_argument('--rollback-kernel-version',
                        default=None,
                        required=False,
                        help='Reset "main" to state prior to autorelease of the given kernel version')

    parser.add_argument('--kernel-repo-path',
                        type=str,
                        default=None,
                        required=False,
                        help='Instead of downloading from git, use existing local repos for autocommits')

    parser.add_argument('--kernel-repo-branch',
                        type=str,
                        default='main',
                        required=False,
                        help='Branch of FreeRTOS Kernel repository to release.')

    parser.add_argument('--use-git-ssh',
                        default=False,
                        action='store_true',
                        help='Use SSH endpoints to interface git remotes, instead of HTTPS')

    parser.add_argument('--unit-test',
                        action='store_true',
                        default=False,
                        help='Run unit tests.')

    parser.add_argument('--do-not-push',
                        action='store_true',
                        default=False,
                        help='Do not push the changes but only make local commits.')

    return parser

def main():
    cmd = configure_argparser()
    args = cmd.parse_args()

    # Auth
    if not args.do_not_push:
        assert 'GITHUB_TOKEN' in os.environ, 'Set env{GITHUB_TOKEN} to an authorized git PAT'
    mGit = Github(os.environ.get('GITHUB_TOKEN'))

    # Unit tests
    if args.unit_test:
        return

    # Create Releases
    if args.new_kernel_version:
        info('Starting kernel release...')
        logIndentPush()
        rel_kernel = KernelRelease(mGit, args.new_kernel_version, args.kernel_commit, git_ssh=args.use_git_ssh,
                                   git_org=args.git_org, repo_path=args.kernel_repo_path, branch=args.kernel_repo_branch,
                                   main_br_version=args.new_kernel_main_br_version, do_not_push=args.do_not_push)
        rel_kernel.autoRelease()
        logIndentPop()

    if args.new_core_version:
        info('Starting core release...')
        logIndentPush()
        rel_freertos = FreertosRelease(mGit, args.new_core_version, args.core_commit, git_ssh=args.use_git_ssh,
                                       git_org=args.git_org, repo_path=args.core_repo_path, branch=args.core_repo_branch,
                                       do_not_push=args.do_not_push)
        rel_freertos.autoRelease()
        logIndentPop()

    # Undo autoreleases
    if args.rollback_kernel_version:
        info('Starting kernel rollback...')
        rel_kernel = KernelRelease(mGit, args.rollback_kernel_version, args.kernel_commit, git_ssh=args.use_git_ssh,
                                   git_org=args.git_org, repo_path=args.kernel_repo_path, branch=args.kernel_repo_branch,
                                   do_not_push=args.do_not_push)
        logIndentPush()
        rel_kernel.restorePriorToRelease()
        logIndentPop()

    if args.rollback_core_version:
        info('Starting core rollback...')
        logIndentPush()
        rel_freertos = FreertosRelease(mGit, args.rollback_core_version, args.core_commit, git_ssh=args.use_git_ssh,
                                       git_org=args.git_org, repo_path=args.core_repo_path, branch=args.core_repo_branch,
                                       do_not_push=args.do_not_push)
        rel_freertos.restorePriorToRelease()
        logIndentPop()

    info('Review script output for any unexpected behaviour.')
    info('Done.')

if __name__ == '__main__':
    main()
