import argparse
import rampds as rs

NAS_ROOT_PATH = "/nas"
RAMP_KITS_NAS_PATH = f"{NAS_ROOT_PATH}/ramp-kits"
RAMP_SETUP_KITS_NAS_PATH = f"{NAS_ROOT_PATH}/ramp-setup-kits"

def setup_ramp_kit(ramp_kit, version, number, use_blend=False):
    """Setup a RAMP kit with specified configuration"""
    # Set feature engineering based on blend option
    if use_blend:
        feature_engineering = "openfe_blend"
        version_name = f"{version}_blend"
    else:
        feature_engineering = "openfe"
        version_name = version
            
    setup_root = RAMP_SETUP_KITS_NAS_PATH
    kit_root = "openfe_new_setup/"
    
    # Use the original setup function
    rs.setup(
        ramp_kit=ramp_kit,
        setup_root=setup_root,
        kit_root=kit_root,
        version=version_name,          
        number=number,
        feature_engineering=feature_engineering,
    )

def main():
    parser = argparse.ArgumentParser(description="Setup RAMP kit with OpenFE feature engineering")
    parser.add_argument("--ramp-kit", type=str, required=True, 
                       help="Name of the RAMP kit (e.g., 'kaggle_wine', 'kaggle_abalone')")
    parser.add_argument("--blend", action="store_true", 
                       help="Use OpenFE blend feature engineering")
    parser.add_argument("--both", action="store_true",
                       help="Run both blend and non-blend versions automatically")
    parser.add_argument("--version", type=str, default="OpenFE_test",
                       help="Version name (default: OpenFE_test)")
    parser.add_argument("--number", type=int, default=0,
                       help="Number (default: 0)")
    
    args = parser.parse_args()

    args.ramp_kit = f"kaggle_{args.ramp_kit}"

    if args.both:
        # Run both versions
        print(f"Setting up {args.ramp_kit} with standard OpenFE...")
        setup_ramp_kit(args.ramp_kit, args.version, args.number, use_blend=False)
        
        print(f"Setting up {args.ramp_kit} with OpenFE blend...")
        setup_ramp_kit(args.ramp_kit, args.version, args.number, use_blend=True)
    else:
        # Run single version based on blend flag
        setup_ramp_kit(args.ramp_kit, args.version, args.number, use_blend=args.blend)

if __name__ == "__main__":
    main()