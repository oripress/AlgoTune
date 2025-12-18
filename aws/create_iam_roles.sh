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

echo "🎉 IAM roles created (or already existed)!"
