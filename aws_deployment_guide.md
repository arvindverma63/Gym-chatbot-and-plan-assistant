# AWS Hosting Guide for FitPax AI

This guide explains how to host your FitPax Pro AI application on Amazon Web Services (AWS).

## Option 1: AWS App Runner (Easiest)
App Runner is a fully managed service that takes your code/container and handles scaling, load balancing, and SSL automatically.

### Prerequisites
- An AWS Account.
- Your code pushed to a GitHub repository.

### Steps
1. **Open AWS Console**: Navigate to **AWS App Runner**.
2. **Create Service**: Click "Create service".
3. **Source**: Select **Source code repository**. Connect your GitHub account and select the `fitpaxproai` repository and the main branch.
4. **Build Settings**:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000`
   - **Port**: 8000
5. **Service Settings**:
   - **CPU & Memory**: Select **1 vCPU / 2 GB RAM** (The AI loads large datasets, so 2GB is recommended).
6. **Deploy**: AWS will provide a URL (e.g., `https://random-id.us-east-1.awsapprunner.com`) where your app is live!

---

## Option 2: AWS EC2 (Full Control)
Use this if you want a dedicated Virtual Machine.

### Steps
1. **Launch Instance**: Go to **EC2** -> **Launch Instance**.
2. **AMI**: Choose **Amazon Linux 2023** (Free Tier eligible).
3. **Instance Type**: Select **t3.small** (2GB RAM). *Note: t2.micro (1GB) might be too small for your large datasets.*
4. **Security Group**: Allow **SSH (22)** and **Custom TCP (8000)** from anywhere.
5. **Connect & Setup**:
   ```bash
   sudo yum update -y
   sudo yum install python3 python3-pip git -y
   git clone <your-repo-link>
   cd fitpaxproai
   pip3 install -r requirements.txt
   ```
6. **Run the App**:
   ```bash
   python3 -m uvicorn api_v2:app --host 0.0.0.0 --port 8000
   ```

---

## Important Configuration Changes
When hosting on AWS, you must update your `static/app.js` to point to the correct API URL if you aren't using relative paths. However, since the FastAPI app serves the HTML, relative paths like `/chat` will work automatically.

### Handling Large Data Files
Ensure your `data/` folder and `GYM.csv` are uploaded to the AWS environment, as the AI depends on these files to function.
