<p align="center">
  <a href="https://algotune.io/">
    <img
      src="https://github.com/oripress/AlgoTune/blob/main/assets/algotune_title_orange.png?raw=true"
      alt="AlgoTune banner"
      width="800"
    />
  </a>
</p>
<p align="center">
  <a href="https://algotune.io/"><strong>Website</strong></a>&nbsp; | &nbsp;
  <a href="https://arxiv.org/abs/2507.15887"><strong>Paper</strong></a>
</p>

How good are language models at coming up with new algorithms? To try to answer this, we built a benchmark, AlgoTune, comprised of 154 widely used math, physics, and computer science functions. For each function, the goal is to write code that produces the same outputs as the original function, while being faster. In addition to the benchmark, we also provide an agent, AlgoTuner, which allows language models to easily optimize code. For contributions to the AlgoTune benchmark, see the [contribution guide](#-contributing-new-problems-to-algotune).

<p align="center">
  <a href="https://algotune.io/">
    <img
      src="https://github.com/oripress/AlgoTune/blob/main/assets/algotune_banner.png?raw=true"
      alt="AlgoTune banner"
      width="800"
    />
  </a>
</p>

---

### ✨ **New:** AlgoTune can now be easily run on AWS with just an OpenRouter API key and AWS credentials. [Try it out!](#running-on-aws-batch-cloud)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -e .  # or, if you prefer conda:
# conda create -n algotune python=3.10
# conda activate algotune
# pip install -e .

# 2. Add your API key
echo "OPENAI_API_KEY=your_key_here" > .env
# OR
echo "CLAUDE_API_KEY=your_key_here" > .env
```

### Run AlgoTune (stand-alone mode)

#### Generate
```bash
# Measure baseline speed for two tasks (100 ms target)
./algotune.sh --standalone generate --target-time-ms 100 --tasks svm
```

#### Run Agent
```bash
# Ask an LM to optimise the same tasks with model "o4-mini"
./algotune.sh --standalone agent o4-mini svm

# View the aggregated speed-up report
cat reports/agent_summary.json
```

### Running on SLURM (cluster)
When `sbatch` is available the launcher auto-detects SLURM. Use the same two-step workflow:

```bash
# One-time image build (only once per cluster)
sudo singularity build algotune.sif slurm/image.def

# Generate baseline datasets for all tasks
./algotune.sh generate --target-time-ms 100

# Run the agent on all tasks
./algotune.sh agent o4-mini

# Results are summarised in:
cat reports/agent_summary.json
```

### Running on AWS Batch (cloud)
Running AlgoTune on AWS is simple and requires only a minimal setup.

#### Prerequisites & Permissions

**AWS CLI** - Install if not already available:
```bash
pip install awscli
```

**AWS IAM Policy** - Create and attach this policy to your IAM user:

1. Go to **IAM → Policies → Create policy**
2. Click **JSON** tab and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity",
        "batch:*",
        "ecr:*",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeVpcs",
        "ec2:DescribeRouteTables",
        "ec2:DescribeVpcEndpoints",
        "ec2:CreateVpcEndpoint",
        "ec2:ModifyVpcEndpoint",
        "ec2:DescribeInstances",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:CreateTags",
        "s3:CreateBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "iam:PassRole",
        "iam:GetRole",
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:GetInstanceProfile",
        "iam:CreateServiceLinkedRole",
        "logs:GetLogEvents",
        "ecs:ListTasks"
      ],
      "Resource": "*"
    }
  ]
}
```

3. Click **Next**, name it (e.g., `AlgoTuneBatchPolicy`)
4. Click **Create policy**
5. Go to **IAM → Users → [your user] → Permissions → Add permissions → Attach policies directly**
6. Search for `AlgoTuneBatchPolicy` and attach it

**If you have restricted permissions**, manually create:
- **S3 bucket**: `algotune-results-{your-account-id}` in your region
- **ECR repository**: `algotune` in your region
- **VPC resources**: Note your subnet ID and security group ID

#### Quick Start

```bash
# One-time setup
./aws/setup-aws.sh        # Interactive AWS configuration

# Launch jobs
./aws/launch-batch.sh     # Interactive: select model and tasks
```

---

## Viewing Results

Extract the best code for each model/task:
```bash
python3 scripts/extract_results_from_logs.py
```

Or generate HTML logs in the style of [AlgoTune.io](https://algotune.io):
```bash
./html/build-html.sh
```

## Evaluating Code Without Running the Agent

You can add code for each task in directories (following the `./results/` structure) and it will be compiled and evaluated. Note that you have to generate the datasets first.

```bash
# Evaluate all models in ./results
./algotune.sh evaluate

# Evaluate specific models
./algotune.sh evaluate --models "Claude Opus 4" "o4-mini"

# View aggregated speedup results
cat reports/evaluate_summary.json
```

## Citation

If you found this work helpful, please consider citing it using the following:

<details>
<summary> AlgoTune citation</summary>

```bibtex
@article{press2025algotune, title={AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?}, 
author={Press, Ori and Amos, Brandon and Zhao, Haoyu and Wu, Yikai and Ainsworth, Samuel K. and Krupke, Dominik and Kidger, Patrick and Sajed, Touqir and Stellato, Bartolomeo and Park, Jisun and Bosch, Nathanael and Meril, Eli and Steppi, Albert and Zharmagambetov, Arman and Zhang, Fangzhao and Perez-Pineiro, David and Mercurio, Alberto and Zhan, Ni and Abramovich, Talor and Lieret, Kilian and Zhang, Hanlin and Huang, Shirley and Bethge, Matthias and Press, Ofir}, 
journal={arXiv preprint arXiv:2507.15887},
year={2025},
 doi={10.48550/arXiv.2507.15887}, 
 url={https://arxiv.org/abs/2507.15887}}
```
</details>


## Questions?
Feel free to write me at me@oripress.com
