#!/usr/bin/env bash
# Delete the Meme Studio stack. The gallery file system (EFS) is kept on purpose; the script prints how to delete it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STACK=${STACK_NAME:-$(sed -n 's/^STACK_NAME=\([A-Za-z0-9-]*\).*/\1/p' "$HERE/deploy.env" 2>/dev/null)}
STACK=${STACK:-meme-studio}
REGION=${AWS_REGION:-$(sed -n 's/^AWS_REGION=\([a-z0-9-]*\).*/\1/p' "$HERE/deploy.env" 2>/dev/null)}
REGION=${REGION:-$(command aws configure get region 2>/dev/null || echo us-east-1)}
aws() { command aws --region "$REGION" "$@"; }
output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null || true
}

aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1 || { echo "no stack $STACK in $REGION"; exit 0; }
echo "This deletes stack $STACK in $REGION: CloudFront, load balancer, Fargate service, image registry, secrets."
echo "The gallery on EFS ($(output FileSystemId)) is kept."
read -r -p "Type the stack name to confirm: " answer
[[ $answer == "$STACK" ]] || { echo "aborted"; exit 1; }

BUCKET=$(output SourceBucketName)
FS=$(output FileSystemId)
[[ -n $BUCKET && $BUCKET != None ]] && aws s3 rm "s3://$BUCKET" --recursive --only-show-errors
aws cloudformation delete-stack --stack-name "$STACK"
echo "deleting (CloudFront takes a few minutes)…"
aws cloudformation wait stack-delete-complete --stack-name "$STACK"
echo "deleted."
if [[ -n $FS && $FS != None ]]; then
  echo
  echo "The gallery file system $FS still exists (a few cents per GB-month). To delete it for good:"
  echo "  aws efs delete-file-system --file-system-id $FS --region $REGION"
fi
