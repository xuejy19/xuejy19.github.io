'use strict';

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const git = (args, cwd = root, capture = false) => execFileSync('git', args, {
  cwd, stdio: capture ? 'pipe' : 'inherit', encoding: 'utf8',
  env: { ...process.env, GIT_SSH_COMMAND: process.env.GIT_SSH_COMMAND || 'ssh -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=1' }
});

// Validate the actual output and push a normal commit on top of the published
// branch. A concurrent remote change makes the push fail rather than overwrite it.
execFileSync(process.execPath, [path.join(__dirname, 'verify-build.cjs')], { cwd: root, stdio: 'inherit' });
const sourceCommit = git(['rev-parse', 'HEAD'], root, true).trim();
git(['fetch', 'origin', 'master']);
const publishDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wind-blog-publish-'));
git(['worktree', 'add', '--detach', publishDir, 'origin/master']);
try {
  for (const name of fs.readdirSync(publishDir)) {
    if (name !== '.git') fs.rmSync(path.join(publishDir, name), { recursive: true, force: true });
  }
  for (const name of fs.readdirSync(path.join(root, 'public'))) {
    if (name === '.git' || name === 'CNAME') throw new Error(`Unexpected publish file: ${name}`);
    fs.cpSync(path.join(root, 'public', name), path.join(publishDir, name), { recursive: true });
  }
  fs.writeFileSync(path.join(publishDir, '.nojekyll'), '');
  fs.writeFileSync(path.join(publishDir, 'build-info.json'), JSON.stringify({
    theme: 'hexo-theme-matery', hexo: require('../package.json').hexo.version,
    sourceCommit, sourceDirty: Boolean(git(['status', '--porcelain'], root, true).trim()),
    publishedAt: new Date().toISOString()
  }, null, 2) + '\n');
  git(['add', '--all'], publishDir);
  git(['commit', '-m', 'Publish restored Matery blog with updated Hexo and local assets'], publishDir);
  git(['push', 'origin', 'HEAD:master'], publishDir);
  console.log(`Published Matery from source ${sourceCommit}.`);
  git(['worktree', 'remove', publishDir]);
} catch (error) {
  console.error(`Publish checkout retained for inspection: ${publishDir}`);
  throw error;
}
