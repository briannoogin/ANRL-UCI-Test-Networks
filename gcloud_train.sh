
TRAINER_PACKAGE_PATH="./experiment"
MAIN_TRAINER_MODULE="experiment.experiment"
PACKAGE_STAGING_PATH="gs://anrl-storage"

now=$(date +"%m%d%Y_%H%M%S")
JOB_NAME="ANRL_experiment_$now"
MODEL_NAME=$JOB_NAME.h5
JOB_DIR="gs://anrl-storage"
REGION="us-central1"
MODE='CLOUD'

gcloud ml-engine jobs submit training $JOB_NAME \
--module-name=$MAIN_TRAINER_MODULE \
--package-path=$TRAINER_PACKAGE_PATH \
--job-dir=$JOB_DIR \
--region=$REGION \
--python-version 3.5 \
--config config.yaml \
--runtime-version 1.13 \
-- \
