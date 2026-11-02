variable "project_id" {
  description = "Google Cloud project with billing enabled and TPU quota in the selected zone."
  type        = string
}

variable "zone" {
  description = "A zone with quota and capacity for the selected eight-chip TPU."
  type        = string
}

variable "tpu_name" {
  description = "Name assigned to the TPU VM."
  type        = string
  default     = "glm-tpu"

  validation {
    condition     = can(regex("^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", var.tpu_name))
    error_message = "tpu_name must be a lowercase GCP resource name (2–63 characters)."
  }
}

variable "accelerator_type" {
  description = "Eight-chip TPU type supported by this repository's recipes."
  type        = string
  default     = "v5litepod-8"

  validation {
    condition     = contains(["v5litepod-8", "v6e-8"], var.accelerator_type)
    error_message = "This recipe requires eight chips: use v5litepod-8 or v6e-8."
  }
}

variable "runtime_version" {
  description = "Cloud TPU runtime matching accelerator_type."
  type        = string
  default     = "v2-alpha-tpuv5-lite"

  validation {
    condition = (
      (var.accelerator_type == "v5litepod-8" && var.runtime_version == "v2-alpha-tpuv5-lite") ||
      (var.accelerator_type == "v6e-8" && var.runtime_version == "v2-alpha-tpuv6e")
    )
    error_message = "Use v2-alpha-tpuv5-lite for v5litepod-8, or v2-alpha-tpuv6e for v6e-8."
  }
}

variable "spot" {
  description = "Use a lower-cost Spot TPU VM. Spot VMs can be reclaimed without notice."
  type        = bool
  default     = true
}

variable "enable_external_ips" {
  description = "Give TPU workers external IPs for outbound downloads and gcloud SSH connectivity."
  type        = bool
  default     = true
}
