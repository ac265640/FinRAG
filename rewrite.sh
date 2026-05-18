set -e
git checkout -b temp_rewrite
git reset --hard 56e8c37

# Rewrite 8f48224
git cherry-pick 8f48224
git commit --amend -m "feat: stabilize user interface layout and optimize backend error handling"

# Rewrite 80f351f
git cherry-pick 80f351f
git commit --amend -m "chore: synchronize workspace configuration and development environment settings"

# Rewrite dcac103
git cherry-pick dcac103
git commit --amend -m "feat: integrate modern landing page and dynamic chat routing"

# Rewrite 827ff27
git cherry-pick 827ff27
git commit --amend -m "chore: remove automated CI workflows"

git checkout main
git reset --hard temp_rewrite
git branch -D temp_rewrite
git push origin main --force
