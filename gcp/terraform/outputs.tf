output "tpu_name" {
  description = "Provisioned TPU VM name."
  value       = google_tpu_v2_vm.recipe.name
}

output "ssh_command" {
  description = "Connect to the TPU VM using the Google Cloud CLI."
  value       = "gcloud compute tpus tpu-vm ssh ${google_tpu_v2_vm.recipe.name} --project=${var.project_id} --zone=${var.zone}"
}

output "runbook" {
  description = "Next step after connecting."
  value       = "From the cloned repository: cd gcp && ./setup.sh, add ~/.kaggle/access_token, ./download.sh, then ./run.sh."
}
