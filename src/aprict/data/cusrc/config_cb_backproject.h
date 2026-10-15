struct config_CB_backproject {
    bool exact_line_integ_backproj;
    bool backproject_all_1s;
    bool mask_to_cylinder;
    float mask_to_cyl_rad;
    bool apply_summand_multiplier;
    float soften_edge_proportion;
    int Z_on_L_factor_power;
    int Z_on_val_factor_power;
    float Z_on_val_factor_val;
    bool approx_mode_direct_raycast;
    bool approx_mode_linear_interpolate;
    unsigned char backproj_weighting_preset;
    unsigned int ssx;
    unsigned int ssy;
    unsigned int ssz;
    float soft_vartheta_v;
    float soft_vartheta_v_softspan;
    float soft_vartheta_h;
    float soft_vartheta_h_softspan;
};

