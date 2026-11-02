# Terraform: create an eight-chip TPU VM

This opt-in Terraform module creates the GCP TPU VM used by the existing
[GCP runbook](../README.md). It provisions compute only. It does not upload
Kaggle credentials, download model weights, expose the serving API, or run
Terraform automatically.

## Requirements

- Terraform 1.5 or later
- A Google Cloud project with billing enabled
- Application Default Credentials (`gcloud auth application-default login`)
- Cloud TPU quota in the selected zone

The recipe needs eight TPU chips. The defaults create a v5e `v5litepod-8`
using Spot capacity. `v6e-8` is also supported when the runtime is changed to
`v2-alpha-tpuv6e`. Four-chip configurations are rejected because the current
GLM recipe shards across eight devices.

## Create and connect

```bash
cd gcp/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit project_id and verify that the selected zone has eight-chip TPU quota.
terraform init
terraform validate
terraform plan
terraform apply
terraform output -raw ssh_command
```

`terraform apply` creates a billable TPU VM. Spot instances can be reclaimed
without notice. When finished, run `terraform destroy` to release the TPU and
stop its compute charges. Terraform does not automatically run `destroy`.

After connecting with the emitted `gcloud` command, follow the existing GCP
runbook to clone the repository, install the pinned runtime, add the Kaggle
credential file with mode `0600`, download datasets, and start the server. The
server binds to loopback; reach it from your workstation through the SSH port
forward in the GCP runbook. Do not open the model API to the public network.

If direct Spot creation fails because the zone has no available capacity, use
the runbook's `gcloud compute tpus queued-resources` flow. The Cloud TPU API
queues requests while capacity is unavailable; Terraform's direct VM resource
does not provide that wait-in-queue behavior.

## State and credentials

Terraform state is local by default and can contain infrastructure metadata.
Keep it private, do not commit `terraform.tfvars` or `*.tfstate`, and configure
a protected remote state backend before sharing state with a team. Google
credentials are obtained through ADC; do not put service-account keys or
Kaggle tokens in Terraform variables, metadata, or state.

The TPU VM resource is provided by the Google Beta provider. Review the plan
and the current [TPU VM resource documentation](https://registry.terraform.io/providers/hashicorp/google-beta/latest/docs/resources/tpu_v2_vm)
before applying it in a production project.
