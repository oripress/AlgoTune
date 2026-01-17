#!/usr/bin/env bash
set -euo pipefail

#######################################################
# AlgoTune AWS Interactive Setup Script
# Configures AWS Batch deployment with minimal manual work
#######################################################

echo "╔═══════════════════════════════════════════════╗"
echo "║  AlgoTune AWS Batch Interactive Setup         ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Check if aws/.env exists
if [ -f "$ENV_FILE" ]; then
  echo "⚠️  Found existing aws/.env file"
  echo ""
  read -p "Do you want to (k)eep it, (r)econfigure, or (m)erge? [k/r/m]: " ENV_CHOICE

  case "$ENV_CHOICE" in
    k|K)
      echo "✓ Keeping existing aws/.env file"
      exit 0
      ;;
    r|R)
      echo "→ Will create new aws/.env file"
      mv "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
      echo "  (Backed up old aws/.env)"
      ;;
    m|M)
      echo "→ Will merge with existing aws/.env"
      source "$ENV_FILE" 2>/dev/null || true
      ;;
    *)
      echo "✓ Keeping existing aws/.env file"
      exit 0
      ;;
  esac
  echo ""
fi

#######################################################
# AWS Credentials
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 1: AWS Credentials"
echo "════════════════════════════════════════════════"
echo ""

# AWS Access Key ID
if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "Current AWS Access Key ID: ${AWS_ACCESS_KEY_ID:0:10}..."
  read -p "Keep this value? [Y/n]: " KEEP_KEY
  if [[ ! "$KEEP_KEY" =~ ^[Nn] ]]; then
    :  # Keep existing
  else
    read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
  fi
else
  read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
fi

# AWS Secret Access Key
if [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "Current AWS Secret Access Key: ****${AWS_SECRET_ACCESS_KEY: -4}"
  read -p "Keep this value? [Y/n]: " KEEP_SECRET
  if [[ ! "$KEEP_SECRET" =~ ^[Nn] ]]; then
    :  # Keep existing
  else
    read -sp "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    echo ""
  fi
else
  read -sp "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
  echo ""
fi

# AWS Region
if [ -n "${AWS_REGION:-}" ]; then
  echo "Current AWS Region: $AWS_REGION"
  read -p "Keep this value? [Y/n]: " KEEP_REGION
  if [[ ! "$KEEP_REGION" =~ ^[Nn] ]]; then
    :  # Keep existing
  else
    read -p "Enter AWS Region [us-east-1]: " AWS_REGION
    AWS_REGION=${AWS_REGION:-us-east-1}
  fi
else
  read -p "Enter AWS Region [us-east-1]: " AWS_REGION
  AWS_REGION=${AWS_REGION:-us-east-1}
fi

echo ""
echo "→ Validating AWS credentials..."

# Export for AWS CLI
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION

# Validate credentials
if ! ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>&1); then
  echo "❌ ERROR: Invalid AWS credentials"
  echo "   $ACCOUNT_ID"
  exit 1
fi

echo "  ✓ Authenticated as AWS Account: $ACCOUNT_ID"
echo ""

#######################################################
# AWS Networking
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 2: AWS Networking"
echo "════════════════════════════════════════════════"
echo ""

# Try to auto-detect default VPC
echo "→ Auto-detecting network configuration..."

DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")

if [ -n "$DEFAULT_VPC" ] && [ "$DEFAULT_VPC" != "None" ]; then
  echo "  ✓ Found default VPC: $DEFAULT_VPC"

  # Get all subnets in default VPC (comma-separated) for multi-AZ capacity
  AWS_SUBNET_ID=${AWS_SUBNET_ID:-$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query "Subnets[].SubnetId" --output text 2>/dev/null | tr '\t' ',' | sed 's/,$//' || echo "")}

  # Get default security group
  AWS_SG_ID=${AWS_SG_ID:-$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")}

  echo "  ✓ Subnet(s): $AWS_SUBNET_ID"
  echo "  ✓ Security Group: $AWS_SG_ID"
  echo ""
  read -p "Use these network settings? [Y/n]: " USE_AUTO

  if [[ "$USE_AUTO" =~ ^[Nn] ]]; then
    read -p "Enter Subnet ID(s) (comma-separated for multiple AZs): " AWS_SUBNET_ID
    read -p "Enter Security Group ID: " AWS_SG_ID
  fi
else
  echo "  ⚠️  Could not auto-detect network configuration"
  echo ""
  echo "Let's find your available subnets and security groups..."
  echo ""

  # List available subnets
  echo "Available Subnets:"
  aws ec2 describe-subnets \
    --query 'Subnets[*].[SubnetId,VpcId,AvailabilityZone,CidrBlock]' \
    --output table 2>/dev/null || echo "  (Unable to list subnets - check AWS permissions)"
  echo ""

  # Prompt for subnet
  if [ -n "${AWS_SUBNET_ID:-}" ]; then
    echo "Current Subnet ID(s): $AWS_SUBNET_ID"
    read -p "Keep this value? [Y/n]: " KEEP_SUBNET
    if [[ "$KEEP_SUBNET" =~ ^[Nn] ]]; then
      read -p "Enter Subnet ID(s) (e.g., subnet-abc123,subnet-def456): " AWS_SUBNET_ID
    fi
  else
    read -p "Enter Subnet ID(s) (e.g., subnet-abc123,subnet-def456): " AWS_SUBNET_ID
  fi

  # Get VPC from subnet to find matching security groups
  if [ -n "$AWS_SUBNET_ID" ]; then
    PRIMARY_SUBNET=$(echo "$AWS_SUBNET_ID" | cut -d',' -f1)
    VPC_ID=$(aws ec2 describe-subnets --subnet-ids "$PRIMARY_SUBNET" --query 'Subnets[0].VpcId' --output text 2>/dev/null || echo "")

    if [ -n "$VPC_ID" ] && [ "$VPC_ID" != "None" ]; then
      echo ""
      echo "Security Groups in VPC $VPC_ID:"
      aws ec2 describe-security-groups \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'SecurityGroups[*].[GroupId,GroupName,Description]' \
        --output table 2>/dev/null || echo "  (Unable to list security groups)"
      echo ""
    fi
  fi

  # Prompt for security group
  if [ -n "${AWS_SG_ID:-}" ]; then
    echo "Current Security Group ID: $AWS_SG_ID"
    read -p "Keep this value? [Y/n]: " KEEP_SG
    if [[ "$KEEP_SG" =~ ^[Nn] ]]; then
      read -p "Enter Security Group ID (e.g., sg-abc123): " AWS_SG_ID
    fi
  else
    read -p "Enter Security Group ID (e.g., sg-abc123): " AWS_SG_ID
  fi
fi

echo ""

#######################################################
# S3 Bucket for Results
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 3: S3 Results Bucket"
echo "════════════════════════════════════════════════"
echo ""

S3_RESULTS_BUCKET="algotune-results-${ACCOUNT_ID}"

echo "→ Creating S3 bucket: s3://$S3_RESULTS_BUCKET"

# Try to create bucket (idempotent)
CREATE_OUTPUT=$(aws s3 mb "s3://$S3_RESULTS_BUCKET" --region "$AWS_REGION" 2>&1)
if echo "$CREATE_OUTPUT" | grep -q "BucketAlreadyOwnedByYou\|make_bucket"; then
  echo "  ✓ S3 bucket ready: s3://$S3_RESULTS_BUCKET"
elif aws s3 ls "s3://$S3_RESULTS_BUCKET" >/dev/null 2>&1; then
  echo "  ✓ S3 bucket already exists: s3://$S3_RESULTS_BUCKET"
elif echo "$CREATE_OUTPUT" | grep -q "AccessDenied"; then
  echo "  ❌ ERROR: IAM user lacks S3 bucket creation permissions"
  echo ""
  echo "  You need to either:"
  echo "    1) Create bucket manually in AWS Console:"
  echo "       - Bucket name: $S3_RESULTS_BUCKET"
  echo "       - Region: $AWS_REGION"
  echo ""
  echo "    2) Add this IAM policy to your user:"
  echo '       {'
  echo '         "Effect": "Allow",'
  echo '         "Action": ["s3:CreateBucket", "s3:PutObject", "s3:GetObject", "s3:ListBucket"],'
  echo "         \"Resource\": [\"arn:aws:s3:::algotune-results-*\", \"arn:aws:s3:::algotune-results-*/*\"]"
  echo '       }'
  echo ""
  read -p "  Have you created the bucket? [y/N]: " BUCKET_CREATED
  if [[ ! "$BUCKET_CREATED" =~ ^[Yy]$ ]]; then
    echo "  ⚠️  Continuing without S3 bucket - jobs will fail without it"
  fi
else
  echo "  ⚠️  Warning: Could not create S3 bucket (continuing anyway)"
  echo "     Error: $CREATE_OUTPUT"
fi

echo ""

#######################################################
# Docker Image
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 4: Docker Image"
echo "════════════════════════════════════════════════"
echo ""

# Use public image by default (users can pull without building)
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/oripress/algotune:latest}"

echo "→ Using Docker image: $DOCKER_IMAGE"
echo ""
echo "  ✓ Using public image (no build required)"
echo ""
echo "  To use a custom image instead:"
echo "    1. Build: ./aws/build-image.sh"
echo "    2. Tag: docker tag algotune:latest ghcr.io/yourusername/algotune:latest"
echo "    3. Push: docker push ghcr.io/yourusername/algotune:latest"
echo "    4. Update DOCKER_IMAGE in aws/.env"
echo ""

#######################################################
# Batch Configuration
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 5: Batch Configuration"
echo "════════════════════════════════════════════════"
echo ""

# Use existing values or defaults
BATCH_JOB_QUEUE_NAME=${BATCH_JOB_QUEUE_NAME:-AlgoTuneQueue}
BATCH_JOB_QUEUE_NAME_SPOT=${BATCH_JOB_QUEUE_NAME_SPOT:-AlgoTuneQueue-spot}
BATCH_JOB_QUEUE_NAME_ONDEMAND=${BATCH_JOB_QUEUE_NAME_ONDEMAND:-AlgoTuneQueue-ondemand}
BATCH_JOB_DEF_NAME=${BATCH_JOB_DEF_NAME:-AlgoTuneJobDef}
BATCH_COMPUTE_ENV_NAME=${BATCH_COMPUTE_ENV_NAME:-AlgoTuneCE-r6a-fixed}
BATCH_COMPUTE_ENV_NAME_SPOT=${BATCH_COMPUTE_ENV_NAME_SPOT:-AlgoTuneCE-spot}
BATCH_COMPUTE_ENV_NAME_ONDEMAND=${BATCH_COMPUTE_ENV_NAME_ONDEMAND:-AlgoTuneCE-ondemand}
BATCH_INSTANCE_TYPES=${BATCH_INSTANCE_TYPES:-r6a.large}
BATCH_MAX_VCPUS=${BATCH_MAX_VCPUS:-8}
BATCH_MIN_VCPUS=${BATCH_MIN_VCPUS:-2}
BATCH_JOB_VCPUS=${BATCH_JOB_VCPUS:-2}
BATCH_JOB_MEMORY_MB=${BATCH_JOB_MEMORY_MB:-15000}
BATCH_SPOT_IAM_FLEET_ROLE=${BATCH_SPOT_IAM_FLEET_ROLE:-arn:aws:iam::${ACCOUNT_ID}:role/aws-ec2-spot-fleet-tagging-role}
BATCH_SPOT_ALLOCATION_STRATEGY=${BATCH_SPOT_ALLOCATION_STRATEGY:-SPOT_CAPACITY_OPTIMIZED}
BATCH_SPOT_BID_PERCENTAGE=${BATCH_SPOT_BID_PERCENTAGE:-100}

echo "Using Batch configuration:"
echo "  Queue: $BATCH_JOB_QUEUE_NAME"
echo "  Spot Queue: $BATCH_JOB_QUEUE_NAME_SPOT"
echo "  On-Demand Queue: $BATCH_JOB_QUEUE_NAME_ONDEMAND"
echo "  Job Definition: $BATCH_JOB_DEF_NAME"
echo "  Compute Environment: $BATCH_COMPUTE_ENV_NAME"
echo "  Spot Compute Environment: $BATCH_COMPUTE_ENV_NAME_SPOT"
echo "  On-Demand Compute Environment: $BATCH_COMPUTE_ENV_NAME_ONDEMAND"
echo "  Instance Type: $BATCH_INSTANCE_TYPES"
echo "  vCPUs: $BATCH_JOB_VCPUS"
echo "  Memory: ${BATCH_JOB_MEMORY_MB}MB"
echo "  Min vCPUs: $BATCH_MIN_VCPUS"
echo "  Spot Fleet Role: $BATCH_SPOT_IAM_FLEET_ROLE"
echo "  Spot Allocation: $BATCH_SPOT_ALLOCATION_STRATEGY"
echo "  Spot Bid %: $BATCH_SPOT_BID_PERCENTAGE"
echo "  Note: Defaults target one job per instance with SLURM-like memory (16GB)."
echo ""

#######################################################
# Write .env File
#######################################################

echo "════════════════════════════════════════════════"
echo "Step 6: Writing Configuration"
echo "════════════════════════════════════════════════"
echo ""

cat > "$ENV_FILE" <<EOF
# AWS Credentials
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
AWS_REGION=$AWS_REGION

# AWS Networking
AWS_SUBNET_ID=$AWS_SUBNET_ID
AWS_SG_ID=$AWS_SG_ID

# S3 Results Bucket
S3_RESULTS_BUCKET=$S3_RESULTS_BUCKET

# Docker Image (DockerHub)
DOCKER_IMAGE=$DOCKER_IMAGE

# AWS Batch Configuration
BATCH_JOB_QUEUE_NAME=$BATCH_JOB_QUEUE_NAME
BATCH_JOB_QUEUE_NAME_SPOT=$BATCH_JOB_QUEUE_NAME_SPOT
BATCH_JOB_QUEUE_NAME_ONDEMAND=$BATCH_JOB_QUEUE_NAME_ONDEMAND
BATCH_JOB_DEF_NAME=$BATCH_JOB_DEF_NAME
BATCH_COMPUTE_ENV_NAME=$BATCH_COMPUTE_ENV_NAME
BATCH_COMPUTE_ENV_NAME_SPOT=$BATCH_COMPUTE_ENV_NAME_SPOT
BATCH_COMPUTE_ENV_NAME_ONDEMAND=$BATCH_COMPUTE_ENV_NAME_ONDEMAND
BATCH_INSTANCE_TYPES=$BATCH_INSTANCE_TYPES
BATCH_MAX_VCPUS=$BATCH_MAX_VCPUS
BATCH_MIN_VCPUS=$BATCH_MIN_VCPUS
BATCH_JOB_VCPUS=$BATCH_JOB_VCPUS
BATCH_JOB_MEMORY_MB=$BATCH_JOB_MEMORY_MB
BATCH_SPOT_IAM_FLEET_ROLE=$BATCH_SPOT_IAM_FLEET_ROLE
BATCH_SPOT_ALLOCATION_STRATEGY=$BATCH_SPOT_ALLOCATION_STRATEGY
BATCH_SPOT_BID_PERCENTAGE=$BATCH_SPOT_BID_PERCENTAGE
EOF

echo "✓ AWS configuration written to aws/.env"
echo ""
echo "Note: API keys should be configured in the root .env file"
echo "      (OPENROUTER_API_KEY, CLAUDE_API_KEY, etc.)"
echo ""

#######################################################
# Next Steps
#######################################################

echo "════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
if [[ "$DOCKER_IMAGE" == ghcr.io/oripress/algotune* ]]; then
  echo "1. Launch batch jobs:"
  echo "   ./aws/launch-batch.sh"
  echo ""
else
  echo "1. Build and push Docker image to DockerHub:"
  echo "   ./aws/build-image.sh"
  echo "   docker tag algotune:latest $DOCKER_IMAGE"
  echo "   docker push $DOCKER_IMAGE"
  echo ""
  echo "2. Launch batch jobs:"
  echo "   ./aws/launch-batch.sh"
  echo ""
fi
