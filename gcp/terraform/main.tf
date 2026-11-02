resource "google_project_service" "tpu_api" {
  project            = var.project_id
  service            = "tpu.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_tpu_v2_vm" "recipe" {
  provider = google-beta

  project          = var.project_id
  name             = var.tpu_name
  zone             = var.zone
  accelerator_type = var.accelerator_type
  runtime_version  = var.runtime_version
  description      = "Eight-chip TPU VM for kaggle-tpu-lab recipes"

  network_config {
    enable_external_ips = var.enable_external_ips
  }

  scheduling_config {
    spot = var.spot
  }

  labels = {
    managed_by = "terraform"
    recipe     = "kaggle-tpu-lab"
  }

  depends_on = [google_project_service.tpu_api, google_project_service.compute_api]
}
