# Deployment Guide for Bottle Gourd Disease Classification

This guide explains how to deploy your Bottle Gourd Disease Classification app to GitHub Pages and Vercel.

## GitHub Pages Deployment

GitHub Pages only supports static content, so we'll deploy a demo version that works without a backend.

### Step 1: Push Your Code to GitHub

1. Make sure your repository is up-to-date on GitHub:
```bash
git add .
git commit -m "Updated for GitHub Pages deployment"
git push origin main
```

### Step 2: Configure GitHub Pages

1. Go to your repository on GitHub
2. Click on "Settings"
3. Scroll down to "GitHub Pages" section
4. Under "Source", select "main" branch
5. Click "Save"
6. Wait a few minutes for deployment
7. Your site will be available at `https://yourusername.github.io/Bottle-Gourd-Disease-Classification/`

## Vercel Deployment

Vercel can host both static sites and serverless functions, but for simplicity, we'll deploy the static version.

### Step 1: Sign Up for Vercel

1. Go to [vercel.com](https://vercel.com)
2. Sign up using your GitHub account

### Step 2: Import Your Repository

1. Click "Add New Project"
2. Connect to your GitHub account if not already connected
3. Select the "Bottle-Gourd-Disease-Classification" repository
4. Configure as follows:
   - Framework Preset: Other
   - Build Command: Leave empty (for static deployment)
   - Output Directory: Leave as default (usually `.`)
   - Install Command: Leave empty
5. Click "Deploy"

### Step 3: Configure Environment Variables (if needed in future)

If you later want to deploy the full Flask application on Vercel:

1. Go to your project settings in Vercel
2. Go to "Environment Variables"
3. Add any required environment variables

## Important Notes

1. The GitHub Pages and static Vercel deployments will only show the UI with demo functionality
2. For the full AI functionality, you'd need to:
   - Deploy to a platform that supports Python (like Heroku, Vercel with Python serverless functions, etc.)
   - Set up proper environment variables
   - Configure requirements properly

## Troubleshooting

If your deployed site doesn't look right:

1. Check browser console for errors (F12)
2. Verify that CSS and JS paths are correct
3. Make sure all files are properly committed to the repository
4. Check if static assets are being served correctly
