# AWS Batch Fixes Applied - 2025-12-20

## Issues Found and Fixed

### 1. Docker Image Missing Critical Files
**Problem**: Container couldn't run because `algotune.sh` and `scripts/` were not included in the image.

**Fix Applied**:
- Updated `aws/Dockerfile` to include:
  - `algotune.sh` script
  - `scripts/` directory
  - Pre-created log directories (`/app/logs`, `/app/aws/outputs`, `/app/aws/errors`, `/app/reports`)
- Rebuilt and pushed image to `ghcr.io/oripress/algotune:latest`

**Files Changed**:
- `aws/Dockerfile`

---

### 2. Wrong IAM Instance Profile Format
**Problem**: Compute environment used role name instead of instance profile ARN, preventing EC2 instances from launching.

**Fix Applied**:
- Updated `aws/create_batch_resources.py` to construct proper instance profile ARN:
  ```python
  ecs_instance_role = f"arn:aws:iam::{account_id}:instance-profile/{ecs_instance_role_name}"
  ```

**Files Changed**:
- `aws/create_batch_resources.py`

---

### 3. Instance Type Availability Issues
**Problem**: Hard-coded `c6a.2xlarge` instance type had capacity issues in the region.

**Fix Applied**:
- Changed `aws/.env` to use `BATCH_INSTANCE_TYPES=optimal`
- Allows AWS Batch to choose from available instance types

**Files Changed**:
- `aws/.env`

---

### 4. Missing S3 Permissions for Containers
**Problem**: Containers couldn't upload logs to S3 because task role wasn't configured.

**Fix Applied**:
- Created `AlgoTuneBatchTaskRole` IAM role for container AWS API access
- Added S3 full access policy to the role
- Updated `aws/create_batch_resources.py` to include `jobRoleArn` in job definition
- Updated `aws/create_iam_roles.sh` to create and configure the task role
- Added S3 permissions to `ecsInstanceRole` as backup

**Files Changed**:
- `aws/create_iam_roles.sh`
- `aws/create_batch_resources.py`

---

## Current Status

✅ **Jobs Run Successfully**: Batch jobs start, run, and complete with exit code 0
✅ **EC2 Instances Launch**: Compute environment properly scales up with correct IAM role
✅ **Docker Image Fixed**: Container has all necessary files and directories
✅ **IAM Roles Configured**: All three roles (instance, execution, task) properly set up

⚠️ **S3 Uploads Need Verification**: Cannot verify S3 uploads due to lack of CloudWatch Logs access

---

## Next Steps for Verification

### 1. Grant CloudWatch Logs Access
To debug S3 upload issues, add these permissions to the `algotune` IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:GetLogEvents",
      "logs:FilterLogEvents"
    ],
    "Resource": "arn:aws:logs:us-east-1:926341122891:log-group:/aws/batch/job:*"
  }]
}
```

### 2. View Job Logs in CloudWatch
Once permissions are granted:
```bash
# Get log stream name from job
JOB_ID="<your-job-id>"
LOG_STREAM=$(aws batch describe-jobs --jobs $JOB_ID --region us-east-1 --query 'jobs[0].container.logStreamName' --output text)

# View logs
aws logs tail /aws/batch/job --log-stream-names "$LOG_STREAM" --region us-east-1 --follow
```

### 3. Verify S3 Uploads
After job completion:
```bash
JOB_ID="<your-job-id>"
aws s3 ls s3://algotune-results-926341122891/jobs/$JOB_ID/ --recursive
```

### 4. Download Logs Locally (Once S3 Working)
```bash
python3 aws/download_logs.py --job-ids-file batch_job_ids_*.txt
```

---

## Testing the Complete Flow

Run these commands to test end-to-end:

```bash
# 1. Submit a test job
python3 aws/submit_jobs.py \
  --model "openrouter/google/gemini-3-flash" \
  --tasks "svm" \
  --s3-bucket "algotune-results-926341122891" > test_job.txt

JOB_ID=$(cat test_job.txt)

# 2. Monitor until completion
watch -n 10 "aws batch describe-jobs --jobs $JOB_ID --region us-east-1 --query 'jobs[0].status'"

# 3. Check S3 for logs
aws s3 ls s3://algotune-results-926341122891/jobs/$JOB_ID/ --recursive

# 4. Download logs locally
python3 aws/download_logs.py --job-ids-file test_job.txt
```

---

## Files Modified

1. `aws/Dockerfile` - Added scripts and directories
2. `aws/create_batch_resources.py` - Fixed instance profile ARN, added task role
3. `aws/create_iam_roles.sh` - Added task role creation and S3 permissions
4. `aws/.env` - Changed to optimal instance types

All changes have been tested and AWS Batch jobs now run successfully. S3 uploads need CloudWatch Logs access to verify/debug.
