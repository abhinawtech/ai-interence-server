use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc};
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::AppError;
use crate::vector::IndexProfile;

/// Status of a reindexing job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReindexJobStatus {
    /// Job is queued and waiting to start
    Queued,
    /// Job is currently running
    Running,
    /// Job completed successfully  
    Completed,
    /// Job failed with error
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job is paused (can be resumed)
    Paused,
}

/// Priority levels for reindexing jobs
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Configuration for a reindexing job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexJobConfig {
    pub job_id: Uuid,
    pub collection_name: String,
    pub target_profile: IndexProfile,
    pub priority: JobPriority,
    pub max_concurrent_threads: usize,
    pub batch_size: usize,
    pub estimated_duration_minutes: u64,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
}

/// Current state and progress of a reindexing job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexJobState {
    pub config: ReindexJobConfig,
    pub status: ReindexJobStatus,
    pub progress_percentage: f64,
    pub processed_items: u64,
    pub total_items: u64,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub current_stage: String,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub resource_usage: ResourceUsage,
}

/// Resource usage tracking for jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_io_mb_per_sec: f64,
    pub network_io_mb_per_sec: f64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            disk_io_mb_per_sec: 0.0,
            network_io_mb_per_sec: 0.0,
        }
    }
}

/// Commands that can be sent to the job manager
#[derive(Debug)]
pub enum JobCommand {
    StartJob(Uuid),
    PauseJob(Uuid),
    ResumeJob(Uuid),
    CancelJob(Uuid),
    UpdateProgress(Uuid, f64, u64),
    SetError(Uuid, String),
    CompleteJob(Uuid),
}

/// Background reindex manager handles job scheduling and execution
pub struct ReindexManager {
    jobs: Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
    job_queue: Arc<RwLock<Vec<Uuid>>>, // Priority-ordered job queue
    command_tx: mpsc::UnboundedSender<JobCommand>,
    command_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<JobCommand>>>>,
    max_concurrent_jobs: usize,
    running_jobs: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
}

impl ReindexManager {
    pub fn new(max_concurrent_jobs: usize) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            job_queue: Arc::new(RwLock::new(Vec::new())),
            command_tx,
            command_rx: Arc::new(RwLock::new(Some(command_rx))),
            max_concurrent_jobs,
            running_jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start the background job processor
    pub async fn start_processor(&self) -> Result<(), AppError> {
        let command_rx = self.command_rx.write().await.take()
            .ok_or_else(|| AppError::VectorOperationError("Processor already started".to_string()))?;

        let jobs_clone = Arc::clone(&self.jobs);
        let queue_clone = Arc::clone(&self.job_queue);
        let running_jobs_clone = Arc::clone(&self.running_jobs);
        let max_concurrent = self.max_concurrent_jobs;

        tokio::spawn(async move {
            Self::process_commands(command_rx, jobs_clone, queue_clone, running_jobs_clone, max_concurrent).await;
        });

        println!("✅ Reindex manager processor started with {} max concurrent jobs", max_concurrent);
        Ok(())
    }

    /// Internal command processor
    async fn process_commands(
        mut command_rx: mpsc::UnboundedReceiver<JobCommand>,
        jobs: Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        job_queue: Arc<RwLock<Vec<Uuid>>>,
        running_jobs: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
        max_concurrent: usize,
    ) {
        while let Some(command) = command_rx.recv().await {
            match command {
                JobCommand::StartJob(job_id) => {
                    let running_count = running_jobs.read().await.len();
                    if running_count < max_concurrent {
                        Self::execute_job(job_id, Arc::clone(&jobs), Arc::clone(&running_jobs)).await;
                    } else {
                        // Add to queue if at capacity
                        job_queue.write().await.push(job_id);
                    }
                }
                JobCommand::PauseJob(job_id) => {
                    Self::pause_job(job_id, &jobs, &running_jobs).await;
                }
                JobCommand::ResumeJob(job_id) => {
                    Self::resume_job(job_id, &jobs).await;
                }
                JobCommand::CancelJob(job_id) => {
                    Self::cancel_job(job_id, &jobs, &running_jobs).await;
                }
                JobCommand::UpdateProgress(job_id, progress, processed) => {
                    Self::update_job_progress(job_id, progress, processed, &jobs).await;
                }
                JobCommand::SetError(job_id, error_msg) => {
                    Self::set_job_error(job_id, error_msg, &jobs).await;
                }
                JobCommand::CompleteJob(job_id) => {
                    Self::complete_job(job_id, &jobs, &running_jobs).await;
                    // Start next job in queue if any
                    Self::start_next_queued_job(&job_queue, &jobs, &running_jobs, max_concurrent).await;
                }
            }
        }
    }

    /// Schedule a new reindexing job
    pub async fn schedule_reindex_job(
        &self,
        collection_name: String,
        target_profile: IndexProfile,
        priority: JobPriority,
    ) -> Result<Uuid, AppError> {
        let job_id = Uuid::new_v4();
        
        let config = ReindexJobConfig {
            job_id,
            collection_name: collection_name.clone(),
            target_profile,
            priority,
            max_concurrent_threads: 4,
            batch_size: 1000,
            estimated_duration_minutes: 10, // Will be updated based on collection size
            created_at: Utc::now(),
            scheduled_at: None,
        };

        let job_state = ReindexJobState {
            config,
            status: ReindexJobStatus::Queued,
            progress_percentage: 0.0,
            processed_items: 0,
            total_items: 0, // Will be determined when job starts
            started_at: None,
            completed_at: None,
            error_message: None,
            current_stage: "Queued".to_string(),
            estimated_completion: None,
            resource_usage: ResourceUsage::default(),
        };

        // Insert job into storage
        self.jobs.write().await.insert(job_id, job_state);
        
        // Add to priority queue
        self.insert_job_in_priority_order(job_id).await;

        println!("📋 Scheduled reindex job {} for collection '{}' with {:?} priority", 
                job_id, collection_name, priority);
        
        Ok(job_id)
    }

    /// Insert job in priority order
    async fn insert_job_in_priority_order(&self, job_id: Uuid) {
        let jobs = self.jobs.read().await;
        let job = jobs.get(&job_id).unwrap();
        let _job_priority = job.config.priority;
        drop(jobs);

        let mut queue = self.job_queue.write().await;
        
        // Find correct position based on priority
        let position = queue.iter().position(|&_id| {
            // This is simplified - in real implementation we'd need to look up priorities
            false // Insert at end for now
        }).unwrap_or(queue.len());
        
        queue.insert(position, job_id);
    }

    /// Start a specific job
    pub async fn start_job(&self, job_id: Uuid) -> Result<(), AppError> {
        self.command_tx.send(JobCommand::StartJob(job_id))
            .map_err(|_| AppError::VectorOperationError("Failed to send start command".to_string()))?;
        Ok(())
    }

    /// Execute a reindexing job
    async fn execute_job(
        job_id: Uuid,
        jobs: Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        running_jobs: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    ) {
        // Update job status to running
        {
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                job.status = ReindexJobStatus::Running;
                job.started_at = Some(Utc::now());
                job.current_stage = "Initializing".to_string();
                job.total_items = 10000; // Simulate collection size
            }
        }

        let jobs_clone = Arc::clone(&jobs);
        
        // Spawn the actual job execution
        let handle = tokio::spawn(async move {
            Self::simulate_reindex_work(job_id, jobs_clone).await;
        });

        // Store the handle for potential cancellation
        running_jobs.write().await.insert(job_id, handle);
        
        println!("🔄 Started reindex job {}", job_id);
    }

    /// Simulate reindexing work with progress updates
    async fn simulate_reindex_work(
        job_id: Uuid,
        jobs: Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
    ) {
        // Simulate multi-stage reindexing process
        let stages = vec![
            ("Analyzing collection", 10),
            ("Backing up current index", 20),
            ("Building new index", 60),
            ("Optimizing index", 80),
            ("Swapping indices", 95),
            ("Verifying integrity", 100),
        ];

        for (stage_name, target_progress) in stages {
            // Check if job is still running (not cancelled/paused)
            {
                let jobs_guard = jobs.read().await;
                if let Some(job) = jobs_guard.get(&job_id) {
                    if job.status != ReindexJobStatus::Running {
                        return; // Job was cancelled or paused
                    }
                }
            }

            // Update current stage
            {
                let mut jobs_guard = jobs.write().await;
                if let Some(job) = jobs_guard.get_mut(&job_id) {
                    job.current_stage = stage_name.to_string();
                    job.progress_percentage = target_progress as f64;
                    job.processed_items = (job.total_items as f64 * target_progress as f64 / 100.0) as u64;
                    
                    // Simulate resource usage
                    job.resource_usage = ResourceUsage {
                        cpu_usage_percent: 45.0 + (target_progress as f64 * 0.3),
                        memory_usage_mb: 256.0 + (target_progress as f64 * 2.0),
                        disk_io_mb_per_sec: 12.5,
                        network_io_mb_per_sec: 8.3,
                    };

                    // Update estimated completion
                    if job.progress_percentage > 0.0 {
                        let elapsed = Utc::now().signed_duration_since(job.started_at.unwrap());
                        let total_estimated = elapsed.num_seconds() as f64 / (job.progress_percentage / 100.0);
                        job.estimated_completion = Some(job.started_at.unwrap() + chrono::Duration::seconds(total_estimated as i64));
                    }
                }
            }

            println!("⚡ Job {} - {}: {}% complete", job_id, stage_name, target_progress);
            
            // Simulate work time
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }

        // Mark job as completed
        {
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                job.status = ReindexJobStatus::Completed;
                job.completed_at = Some(Utc::now());
                job.progress_percentage = 100.0;
                job.current_stage = "Completed".to_string();
            }
        }

        println!("✅ Reindex job {} completed successfully", job_id);
    }

    /// Pause a running job
    async fn pause_job(
        job_id: Uuid,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        running_jobs: &Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    ) {
        // Update status
        {
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                job.status = ReindexJobStatus::Paused;
            }
        }

        // Abort the running task
        if let Some(handle) = running_jobs.write().await.remove(&job_id) {
            handle.abort();
            println!("⏸️ Paused reindex job {}", job_id);
        }
    }

    /// Resume a paused job
    async fn resume_job(
        job_id: Uuid,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
    ) {
        let mut jobs_guard = jobs.write().await;
        if let Some(job) = jobs_guard.get_mut(&job_id) {
            if job.status == ReindexJobStatus::Paused {
                job.status = ReindexJobStatus::Queued; // Will be picked up by processor
                println!("▶️ Resumed reindex job {}", job_id);
            }
        }
    }

    /// Cancel a job
    async fn cancel_job(
        job_id: Uuid,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        running_jobs: &Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    ) {
        // Update status
        {
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                job.status = ReindexJobStatus::Cancelled;
                job.completed_at = Some(Utc::now());
            }
        }

        // Abort running task if any
        if let Some(handle) = running_jobs.write().await.remove(&job_id) {
            handle.abort();
        }
        
        println!("❌ Cancelled reindex job {}", job_id);
    }

    /// Update job progress
    async fn update_job_progress(
        job_id: Uuid,
        progress: f64,
        processed: u64,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
    ) {
        let mut jobs_guard = jobs.write().await;
        if let Some(job) = jobs_guard.get_mut(&job_id) {
            job.progress_percentage = progress;
            job.processed_items = processed;
        }
    }

    /// Set job error
    async fn set_job_error(
        job_id: Uuid,
        error_msg: String,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
    ) {
        let mut jobs_guard = jobs.write().await;
        if let Some(job) = jobs_guard.get_mut(&job_id) {
            job.status = ReindexJobStatus::Failed;
            job.error_message = Some(error_msg);
            job.completed_at = Some(Utc::now());
        }
    }

    /// Complete a job
    async fn complete_job(
        job_id: Uuid,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        running_jobs: &Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    ) {
        // Remove from running jobs
        running_jobs.write().await.remove(&job_id);
        
        // Status should already be updated by the job itself
        println!("🏁 Job {} finished execution", job_id);
    }

    /// Start next queued job if capacity allows
    async fn start_next_queued_job(
        job_queue: &Arc<RwLock<Vec<Uuid>>>,
        jobs: &Arc<RwLock<HashMap<Uuid, ReindexJobState>>>,
        running_jobs: &Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
        max_concurrent: usize,
    ) {
        let running_count = running_jobs.read().await.len();
        if running_count < max_concurrent {
            if let Some(next_job_id) = job_queue.write().await.pop() {
                Self::execute_job(next_job_id, Arc::clone(jobs), Arc::clone(running_jobs)).await;
            }
        }
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: Uuid) -> Option<ReindexJobState> {
        self.jobs.read().await.get(&job_id).cloned()
    }

    /// Get all jobs
    pub async fn get_all_jobs(&self) -> HashMap<Uuid, ReindexJobState> {
        self.jobs.read().await.clone()
    }

    /// Get jobs by status
    pub async fn get_jobs_by_status(&self, status: ReindexJobStatus) -> Vec<ReindexJobState> {
        self.jobs.read().await
            .values()
            .filter(|job| job.status == status)
            .cloned()
            .collect()
    }

    /// Get queue status
    pub async fn get_queue_status(&self) -> (usize, usize, usize) {
        let queue_size = self.job_queue.read().await.len();
        let running_count = self.running_jobs.read().await.len();
        let total_jobs = self.jobs.read().await.len();
        
        (queue_size, running_count, total_jobs)
    }

    /// Pause job by ID
    pub async fn pause_job_by_id(&self, job_id: Uuid) -> Result<(), AppError> {
        self.command_tx.send(JobCommand::PauseJob(job_id))
            .map_err(|_| AppError::VectorOperationError("Failed to send pause command".to_string()))?;
        Ok(())
    }

    /// Resume job by ID
    pub async fn resume_job_by_id(&self, job_id: Uuid) -> Result<(), AppError> {
        self.command_tx.send(JobCommand::ResumeJob(job_id))
            .map_err(|_| AppError::VectorOperationError("Failed to send resume command".to_string()))?;
        Ok(())
    }

    /// Cancel job by ID
    pub async fn cancel_job_by_id(&self, job_id: Uuid) -> Result<(), AppError> {
        self.command_tx.send(JobCommand::CancelJob(job_id))
            .map_err(|_| AppError::VectorOperationError("Failed to send cancel command".to_string()))?;
        Ok(())
    }
}

/// Factory function to create reindex manager
pub fn create_reindex_manager(max_concurrent_jobs: usize) -> ReindexManager {
    ReindexManager::new(max_concurrent_jobs)
}