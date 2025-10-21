#!/bin/bash
# filepath: run_openfe_setup.sh

# Configuration
VERSION="OpenFE_test"
NUMBER=0

# Example list of RAMP kits to process
datasets=(
    "concrete_strength"
    "rainfall"
    "wine"
    "attrition"
    "blueberry"
    "heat_flux_fi"
    "abalone"
    "mohs_hardness"
    "crab_age"
    "housing_california"
    "influencers"
    "cirrhosis"
    "sticker"
    "reservations"
    "obesity"
    "loan_approval"
    "calories"
    "credit_fusion"
    "churn"
    "failure"
    "unknown_a"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting OpenFE setup for ${#datasets[@]} datasets${NC}"
echo "Run both versions: $RUN_BOTH"
if [ "$RUN_BOTH" = false ]; then
    echo "Blend mode: $USE_BLEND"
fi
echo "Version: $VERSION"
echo "Number: $NUMBER"
echo "----------------------------------------"

# Function to run setup for a single dataset
run_setup() {
    local dataset=$1
    echo -e "${YELLOW}Processing: $dataset${NC}"
    
    # Build command
    cmd="python test_openfe_in_setup.py --ramp-kit $dataset --version $VERSION --number $NUMBER"
    
    # Execute command
    if eval $cmd; then
        echo -e "${GREEN}✓ Successfully processed: $dataset${NC}"
    else
        echo -e "${RED}✗ Failed to process: $dataset${NC}"
        return 1
    fi
    echo "----------------------------------------"
}

# Main execution
failed_datasets=()
successful_datasets=()

for dataset in "${datasets[@]}"; do
    if run_setup "$dataset"; then
        successful_datasets+=("$dataset")
    else
        failed_datasets+=("$dataset")
    fi
done

# Summary
echo -e "${YELLOW}Summary:${NC}"
echo "Total datasets: ${#datasets[@]}"
echo -e "${GREEN}Successful: ${#successful_datasets[@]}${NC}"
if [ ${#successful_datasets[@]} -gt 0 ]; then
    printf '  - %s\n' "${successful_datasets[@]}"
fi

echo -e "${RED}Failed: ${#failed_datasets[@]}${NC}"
if [ ${#failed_datasets[@]} -gt 0 ]; then
    printf '  - %s\n' "${failed_datasets[@]}"
fi

# Exit with error code if any failed
if [ ${#failed_datasets[@]} -gt 0 ]; then
    exit 1
fi

echo -e "${GREEN}All datasets processed successfully!${NC}"