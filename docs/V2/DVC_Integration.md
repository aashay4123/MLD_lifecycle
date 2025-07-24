# DVC Integration for MLD Lifecycle

### **Step 1: Initialize DVC in your existing Git-tracked repo**

```bash
cd MLD_lifecycle  # Or your actual root
git init          # If not already done
dvc init
```

---

### **Step 2: Untrack data folders from Git and add to DVC instead**

Let’s say your `Data/`, `mlruns/`, `reports/`, `.zen/`, `artifacts/` and any EDA or drift folders are the candidates:

```bash
# Stop Git from tracking
echo 'Data/' >> .gitignore
echo 'artifacts/' >> .gitignore
echo 'mlruns/' >> .gitignore
echo '.zen/' >> .gitignore
echo 'reports/' >> .gitignore

# Now track with DVC
dvc add Data/ -o dvc/Data.dvc
dvc add artifacts/ -o dvc/artifacts.dvc
dvc add mlruns/ -o dvc/mlruns.dvc
dvc add .zen/ -o dvc/.zen.dvc
dvc add reports/ -o dvc/reports.dvc

# Commit changes
git add .gitignore dvc/Data.dvc dvc/artifacts.dvc dvc/mlruns.dvc dvc/.zen.dvc dvc/reports.dvc
git commit -m "Initialize DVC with data tracking"
```

---

### **Step 3: Set up DVC remote**

By default, set up a local remote:

```bash
dvc remote add -d local_cache .dvc/cache
dvc remote modify local_cache type local
```

Optionally, configure an S3 remote:

```bash
# Optional: Add S3 remote (commented out in config)
# dvc remote add s3remote s3://your-bucket-name/path
# dvc remote modify s3remote access_key_id <YOUR_KEY>
# dvc remote modify s3remote secret_access_key <YOUR_SECRET>
# dvc remote default s3remote
```

---

### **✅ Usage Instructions**

```bash
# Track changes
dvc repro                     # Reproduces full pipeline
dvc status                   # Check which stages need update
dvc push                     # Push data to remote cache (S3/local)
dvc pull                     # Pull data from cache (when cloning repo)

```

To serve a model:

- You should register and serve using `MLflow` (or FastAPI), but all artifacts used by ZenML can be versioned through this DVC pipeline.
