import argparse
import rampds as rs

NAS_ROOT_PATH = "/nas"
RAMP_KITS_NAS_PATH = f"{NAS_ROOT_PATH}/ramp-kits"
RAMP_SETUP_KITS_NAS_PATH = f"{NAS_ROOT_PATH}/ramp-setup-kits"

def main():
    parser = argparse.ArgumentParser(description="Setup RAMP kit with OpenFE feature engineering")
    parser.add_argument("--ramp-kit", type=str, required=True, 
                       help="Name of the RAMP kit (e.g., 'kaggle_wine', 'kaggle_abalone')")
    parser.add_argument("--version", type=str, default="OpenFE_test",
                       help="Version name (default: OpenFE_test)")
    parser.add_argument("--number", type=int, default=0,
                       help="Number (default: 0)")
    
    args = parser.parse_args()

    args.ramp_kit = f"kaggle_{args.ramp_kit}"

    setup_root = RAMP_SETUP_KITS_NAS_PATH
    kit_root = "openfe_new_setup/"
    
    # Use the original setup function
    rs.setup(
        ramp_kit=args.ramp_kit,
        setup_root=setup_root,
        kit_root=kit_root,
        version=args.version,
        number=args.number,
        feature_engineering="openfe",
    )

if __name__ == "__main__":
    main()