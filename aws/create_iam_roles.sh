#!/usr/bin/env bash
set -e

# Load AWS credentials from .env
export $(cat .env | xargs)

echo "🛠 Creating IAM roles required for AWS Batch ..."

############################
# Create ecsInstanceRole
############################

echo "📌 Setting up ecsInstanceRole (for Batch EC2 compute nodes)..."

cat > /tmp/ecsInstanceRole-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name ecsInstanceRole \
    --assume-role-policy-document file:///tmp/ecsInstanceRole-trust.json \
    2>/dev/null && echo "Created ecsInstanceRole" || echo "ecsInstanceRole already exists"

echo "Attaching AmazonEC2ContainerServiceforEC2Role policy..."
aws iam attach-role-policy \
    --role-name ecsInstanceRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role

echo "Attaching S3 access policy to ecsInstanceRole (for logs upload)..."
aws iam put-role-policy \
    --role-name ecsInstanceRole \
    --policy-name AlgoTuneS3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::algotune-results-*/*",
                "arn:aws:s3:::algotune-results-*"
            ]
        }]
    }' 2>/dev/null && echo "S3 access policy attached to ecsInstanceRole" || echo "S3 policy already exists"

echo "Creating instance profile..."
aws iam create-instance-profile \
    --instance-profile-name ecsInstanceRole \
    2>/dev/null && echo "Created instance profile" || echo "Instance profile already exists"

echo "Adding role to instance profile..."
aws iam add-role-to-instance-profile \
    --instance-profile-name ecsInstanceRole \
    --role-name ecsInstanceRole \
    2>/dev/null && echo "Role added to instance profile" || echo "Role already in instance profile"

echo "✔ ecsInstanceRole ready"

############################
# Create ecsTaskExecutionRole
############################

echo "📌 Setting up ecsTaskExecutionRole (for Batch job execution)..."

cat > /tmp/ecsTaskExecutionRole-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ecs-tasks.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file:///tmp/ecsTaskExecutionRole-trust.json \
    2>/dev/null && echo "Created ecsTaskExecutionRole" || echo "ecsTaskExecutionRole already exists"

echo "Attaching AmazonECSTaskExecutionRolePolicy policy..."
aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

echo "Attaching S3 access policy for results upload..."
aws iam put-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-name AlgoTuneS3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::algotune-results-*/*",
                "arn:aws:s3:::algotune-results-*"
            ]
        }]
    }' 2>/dev/null && echo "S3 access policy attached" || echo "S3 policy already exists or failed to attach"

echo "✔ ecsTaskExecutionRole ready"

############################
# Create AlgoTuneBatchTaskRole (for container AWS API access)
############################

echo "📌 Setting up AlgoTuneBatchTaskRole (for container S3 access)..."

cat > /tmp/AlgoTuneBatchTaskRole-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ecs-tasks.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
    --role-name AlgoTuneBatchTaskRole \
    --assume-role-policy-document file:///tmp/AlgoTuneBatchTaskRole-trust.json \
    2>/dev/null && echo "Created AlgoTuneBatchTaskRole" || echo "AlgoTuneBatchTaskRole already exists"

echo "Attaching S3 access policy..."
aws iam put-role-policy \
    --role-name AlgoTuneBatchTaskRole \
    --policy-name S3FullAccess \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:*"],
            "Resource": [
                "arn:aws:s3:::algotune-results-*/*",
                "arn:aws:s3:::algotune-results-*"
            ]
        }]
    }' 2>/dev/null && echo "S3 access policy attached" || echo "S3 policy already exists"

echo "✔ AlgoTuneBatchTaskRole ready"

echo "🎉 IAM roles created (or already existed)!"
