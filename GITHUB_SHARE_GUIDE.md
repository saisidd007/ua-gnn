# GitHub sharing guide

This project is ready to be shared on GitHub as a source-code repository. Use the following steps on your machine to publish it cleanly.

## 1) Install Git

If Git is not installed on your PC, install it from:

https://git-scm.com/downloads

## 2) Open PowerShell in the project folder

```powershell
cd "C:\Users\rockk\OneDrive\Desktop\traffic-flow-gnn"
```

## 3) Initialize the repository

```powershell
git init
git add .
git commit -m "Initial project upload"
```

## 4) Create the GitHub repo

Go to GitHub and create a new repository.

- Keep it private if you want to share only with your professor.
- Choose a clear name such as `traffic-flow-gnn`

## 5) Connect local repo to GitHub

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/traffic-flow-gnn.git
git push -u origin main
```

## 6) Share with professor

- Send the GitHub repository link
- Or invite the professor as a collaborator if it is a private repository
- If the professor needs dataset/model files, upload them to OneDrive/Google Drive and include the link in the README

## Notes

The repository intentionally excludes large generated files like model checkpoints, data folders, and heavy outputs so the GitHub repo stays clean and easy to review.

If you want to include the dataset later, keep it in a separate cloud link instead of pushing it directly to GitHub.
