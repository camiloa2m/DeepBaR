#!/bin/bash

# Main summary log
SUMMARY_LOG="execution_summary.txt"
echo "Execution summary - $(date)" > $SUMMARY_LOG
echo "--------------------------------------" >> $SUMMARY_LOG

# List of Python commands to run
commands=(
    "python attack_on_finetuning_resnet50.py --attack True  --fprob 0.75 --epochs 1"
    "python attack_on_finetuning_resnet50.py --attack True  --fprob 0.50 --epochs 1"
    "python attack_on_finetuninng_resnet50.py --attack True  --fprob 0.25 --epochs 1"
)

# Loop over commands
for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    OUTPUT_LOG="run_$((i+1))_output.log"

    # Record start timestamp
    START_TS=$(date "+%Y-%m-%d %H:%M:%S")
    START_SEC=$(date +%s)

    # Write header to output log
    {
        echo "=== Command $((i+1)) ==="
        echo "Start time: $START_TS"
        echo "Command: $cmd"
        echo "--------------------------------------"
    } > "$OUTPUT_LOG"

    echo "Running command $((i+1)) at $START_TS: $cmd" | tee -a $SUMMARY_LOG
    echo "Output logged to $OUTPUT_LOG" | tee -a $SUMMARY_LOG

    # Run command with real-time timestamps and append to output log
    $cmd > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee -a "$OUTPUT_LOG")

    # Capture Python exit code
    STATUS=${PIPESTATUS[0]}

    # Record end timestamp
    END_TS=$(date "+%Y-%m-%d %H:%M:%S")
    END_SEC=$(date +%s)
    DURATION=$((END_SEC - START_SEC))

    # Write summary info to output log and main summary log
    {
        echo "--------------------------------------"
        if [ $STATUS -eq 0 ]; then
            echo "Finished successfully at $END_TS (Duration: ${DURATION}s)"
        else
            echo "Failed with exit code $STATUS at $END_TS (Duration: ${DURATION}s)"
        fi
        echo "======================================"
    } | tee -a "$OUTPUT_LOG" "$SUMMARY_LOG"
done
