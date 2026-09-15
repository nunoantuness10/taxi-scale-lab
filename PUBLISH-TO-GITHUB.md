# Upload from macOS

```bash
git init -b main
git add .
git status
git commit -m "feat: add CPU dataframe benchmark and taxi fare study"
gh repo create taxi-scale-lab --public --source=. --remote=origin --push
gh repo view --web
```

Review `git status` before committing. Raw taxi files, local environments, and `results/` are ignored.
