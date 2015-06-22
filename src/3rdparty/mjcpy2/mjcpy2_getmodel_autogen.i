    out["nq"] = m_model->nq;
    out["nv"] = m_model->nv;
    out["nu"] = m_model->nu;
    out["na"] = m_model->na;
    out["nbody"] = m_model->nbody;
    out["njnt"] = m_model->njnt;
    out["ngeom"] = m_model->ngeom;
    out["nsite"] = m_model->nsite;
    out["ncam"] = m_model->ncam;
    out["nmesh"] = m_model->nmesh;
    out["nmeshvert"] = m_model->nmeshvert;
    out["nmeshface"] = m_model->nmeshface;
    out["nmeshgraph"] = m_model->nmeshgraph;
    out["nhfield"] = m_model->nhfield;
    out["nhfielddata"] = m_model->nhfielddata;
    out["ncpair"] = m_model->ncpair;
    out["neq"] = m_model->neq;
    out["ntendon"] = m_model->ntendon;
    out["nwrap"] = m_model->nwrap;
    out["nnumeric"] = m_model->nnumeric;
    out["nnumericdata"] = m_model->nnumericdata;
    out["ntext"] = m_model->ntext;
    out["ntextdata"] = m_model->ntextdata;
    out["nuser_body"] = m_model->nuser_body;
    out["nuser_jnt"] = m_model->nuser_jnt;
    out["nuser_geom"] = m_model->nuser_geom;
    out["nuser_site"] = m_model->nuser_site;
    out["nuser_eq"] = m_model->nuser_eq;
    out["nuser_tendon"] = m_model->nuser_tendon;
    out["nuser_actuator"] = m_model->nuser_actuator;
    out["nnames"] = m_model->nnames;
    out["nM"] = m_model->nM;
    out["nemax"] = m_model->nemax;
    out["nlmax"] = m_model->nlmax;
    out["ncmax"] = m_model->ncmax;
    out["njmax"] = m_model->njmax;
    out["nctotmax"] = m_model->nctotmax;
    out["nstack"] = m_model->nstack;
    out["nuserdata"] = m_model->nuserdata;
    out["nbuffer"] = m_model->nbuffer;
    out["qpos0"] = toNdarray2<mjtNum>(m_model->qpos0, m_model->nq, 1);
    out["qpos_spring"] = toNdarray2<mjtNum>(m_model->qpos_spring, m_model->nq, 1);
    out["body_parentid"] = toNdarray2<int>(m_model->body_parentid, m_model->nbody, 1);
    out["body_rootid"] = toNdarray2<int>(m_model->body_rootid, m_model->nbody, 1);
    out["body_weldid"] = toNdarray2<int>(m_model->body_weldid, m_model->nbody, 1);
    out["body_jntnum"] = toNdarray2<int>(m_model->body_jntnum, m_model->nbody, 1);
    out["body_jntadr"] = toNdarray2<int>(m_model->body_jntadr, m_model->nbody, 1);
    out["body_dofnum"] = toNdarray2<int>(m_model->body_dofnum, m_model->nbody, 1);
    out["body_dofadr"] = toNdarray2<int>(m_model->body_dofadr, m_model->nbody, 1);
    out["body_geomnum"] = toNdarray2<int>(m_model->body_geomnum, m_model->nbody, 1);
    out["body_geomadr"] = toNdarray2<int>(m_model->body_geomadr, m_model->nbody, 1);
    out["body_gravcomp"] = toNdarray2<mjtByte>(m_model->body_gravcomp, m_model->nbody, 1);
    out["body_pos"] = toNdarray2<mjtNum>(m_model->body_pos, m_model->nbody, 3);
    out["body_quat"] = toNdarray2<mjtNum>(m_model->body_quat, m_model->nbody, 4);
    out["body_ipos"] = toNdarray2<mjtNum>(m_model->body_ipos, m_model->nbody, 3);
    out["body_iquat"] = toNdarray2<mjtNum>(m_model->body_iquat, m_model->nbody, 4);
    out["body_mass"] = toNdarray2<mjtNum>(m_model->body_mass, m_model->nbody, 1);
    out["body_inertia"] = toNdarray2<mjtNum>(m_model->body_inertia, m_model->nbody, 3);
    out["body_invweight0"] = toNdarray2<mjtNum>(m_model->body_invweight0, m_model->nbody, 2);
    out["body_user"] = toNdarray2<mjtNum>(m_model->body_user, m_model->nbody, m_model->nuser_body);
    out["jnt_type"] = toNdarray2<int>(m_model->jnt_type, m_model->njnt, 1);
    out["jnt_qposadr"] = toNdarray2<int>(m_model->jnt_qposadr, m_model->njnt, 1);
    out["jnt_dofadr"] = toNdarray2<int>(m_model->jnt_dofadr, m_model->njnt, 1);
    out["jnt_bodyid"] = toNdarray2<int>(m_model->jnt_bodyid, m_model->njnt, 1);
    out["jnt_islimited"] = toNdarray2<mjtByte>(m_model->jnt_islimited, m_model->njnt, 1);
    out["jnt_pos"] = toNdarray2<mjtNum>(m_model->jnt_pos, m_model->njnt, 3);
    out["jnt_axis"] = toNdarray2<mjtNum>(m_model->jnt_axis, m_model->njnt, 3);
    out["jnt_stiffness"] = toNdarray2<mjtNum>(m_model->jnt_stiffness, m_model->njnt, 1);
    out["jnt_range"] = toNdarray2<mjtNum>(m_model->jnt_range, m_model->njnt, 2);
    out["jnt_compliance"] = toNdarray2<mjtNum>(m_model->jnt_compliance, m_model->njnt, 1);
    out["jnt_timeconst"] = toNdarray2<mjtNum>(m_model->jnt_timeconst, m_model->njnt, 1);
    out["jnt_mindist"] = toNdarray2<mjtNum>(m_model->jnt_mindist, m_model->njnt, 1);
    out["jnt_user"] = toNdarray2<mjtNum>(m_model->jnt_user, m_model->njnt, m_model->nuser_jnt);
    out["dof_bodyid"] = toNdarray2<int>(m_model->dof_bodyid, m_model->nv, 1);
    out["dof_jntid"] = toNdarray2<int>(m_model->dof_jntid, m_model->nv, 1);
    out["dof_parentid"] = toNdarray2<int>(m_model->dof_parentid, m_model->nv, 1);
    out["dof_Madr"] = toNdarray2<int>(m_model->dof_Madr, m_model->nv, 1);
    out["dof_isfrictional"] = toNdarray2<mjtByte>(m_model->dof_isfrictional, m_model->nv, 1);
    out["dof_armature"] = toNdarray2<mjtNum>(m_model->dof_armature, m_model->nv, 1);
    out["dof_damping"] = toNdarray2<mjtNum>(m_model->dof_damping, m_model->nv, 1);
    out["dof_frictionloss"] = toNdarray2<mjtNum>(m_model->dof_frictionloss, m_model->nv, 1);
    out["dof_maxvel"] = toNdarray2<mjtNum>(m_model->dof_maxvel, m_model->nv, 1);
    out["dof_invweight0"] = toNdarray2<mjtNum>(m_model->dof_invweight0, m_model->nv, 1);
    out["geom_type"] = toNdarray2<int>(m_model->geom_type, m_model->ngeom, 1);
    out["geom_contype"] = toNdarray2<int>(m_model->geom_contype, m_model->ngeom, 1);
    out["geom_conaffinity"] = toNdarray2<int>(m_model->geom_conaffinity, m_model->ngeom, 1);
    out["geom_condim"] = toNdarray2<int>(m_model->geom_condim, m_model->ngeom, 1);
    out["geom_bodyid"] = toNdarray2<int>(m_model->geom_bodyid, m_model->ngeom, 1);
    out["geom_dataid"] = toNdarray2<int>(m_model->geom_dataid, m_model->ngeom, 1);
    out["geom_group"] = toNdarray2<int>(m_model->geom_group, m_model->ngeom, 1);
    out["geom_size"] = toNdarray2<mjtNum>(m_model->geom_size, m_model->ngeom, 3);
    out["geom_rbound"] = toNdarray2<mjtNum>(m_model->geom_rbound, m_model->ngeom, 1);
    out["geom_pos"] = toNdarray2<mjtNum>(m_model->geom_pos, m_model->ngeom, 3);
    out["geom_quat"] = toNdarray2<mjtNum>(m_model->geom_quat, m_model->ngeom, 4);
    out["geom_friction"] = toNdarray2<mjtNum>(m_model->geom_friction, m_model->ngeom, 3);
    out["geom_compliance"] = toNdarray2<mjtNum>(m_model->geom_compliance, m_model->ngeom, 1);
    out["geom_timeconst"] = toNdarray2<mjtNum>(m_model->geom_timeconst, m_model->ngeom, 1);
    out["geom_mindist"] = toNdarray2<mjtNum>(m_model->geom_mindist, m_model->ngeom, 1);
    out["geom_user"] = toNdarray2<mjtNum>(m_model->geom_user, m_model->ngeom, m_model->nuser_geom);
    out["geom_rgba"] = toNdarray2<float>(m_model->geom_rgba, m_model->ngeom, 4);
    out["site_bodyid"] = toNdarray2<int>(m_model->site_bodyid, m_model->nsite, 1);
    out["site_group"] = toNdarray2<int>(m_model->site_group, m_model->nsite, 1);
    out["site_pos"] = toNdarray2<mjtNum>(m_model->site_pos, m_model->nsite, 3);
    out["site_quat"] = toNdarray2<mjtNum>(m_model->site_quat, m_model->nsite, 4);
    out["site_user"] = toNdarray2<mjtNum>(m_model->site_user, m_model->nsite, m_model->nuser_site);
    out["mesh_faceadr"] = toNdarray2<int>(m_model->mesh_faceadr, m_model->nmesh, 1);
    out["mesh_facenum"] = toNdarray2<int>(m_model->mesh_facenum, m_model->nmesh, 1);
    out["mesh_vertadr"] = toNdarray2<int>(m_model->mesh_vertadr, m_model->nmesh, 1);
    out["mesh_vertnum"] = toNdarray2<int>(m_model->mesh_vertnum, m_model->nmesh, 1);
    out["mesh_graphadr"] = toNdarray2<int>(m_model->mesh_graphadr, m_model->nmesh, 1);
    out["mesh_vert"] = toNdarray2<float>(m_model->mesh_vert, m_model->nmeshvert, 3);
    out["mesh_face"] = toNdarray2<int>(m_model->mesh_face, m_model->nmeshface, 3);
    out["mesh_graph"] = toNdarray2<int>(m_model->mesh_graph, m_model->nmeshgraph, 1);
    out["hfield_nrow"] = toNdarray2<int>(m_model->hfield_nrow, m_model->nhfield, 1);
    out["hfield_ncol"] = toNdarray2<int>(m_model->hfield_ncol, m_model->nhfield, 1);
    out["hfield_adr"] = toNdarray2<int>(m_model->hfield_adr, m_model->nhfield, 1);
    out["hfield_geomid"] = toNdarray2<int>(m_model->hfield_geomid, m_model->nhfield, 1);
    out["hfield_dir"] = toNdarray2<mjtByte>(m_model->hfield_dir, m_model->nhfield, 1);
    out["hfield_data"] = toNdarray2<float>(m_model->hfield_data, m_model->nhfielddata, 1);
    out["pair_dim"] = toNdarray2<int>(m_model->pair_dim, m_model->ncpair, 1);
    out["pair_geom1"] = toNdarray2<int>(m_model->pair_geom1, m_model->ncpair, 1);
    out["pair_geom2"] = toNdarray2<int>(m_model->pair_geom2, m_model->ncpair, 1);
    out["pair_compliance"] = toNdarray2<mjtNum>(m_model->pair_compliance, m_model->ncpair, 1);
    out["pair_timeconst"] = toNdarray2<mjtNum>(m_model->pair_timeconst, m_model->ncpair, 1);
    out["pair_mindist"] = toNdarray2<mjtNum>(m_model->pair_mindist, m_model->ncpair, 1);
    out["pair_friction"] = toNdarray2<mjtNum>(m_model->pair_friction, m_model->ncpair, 5);
    out["eq_type"] = toNdarray2<int>(m_model->eq_type, m_model->neq, 1);
    out["eq_obj1type"] = toNdarray2<int>(m_model->eq_obj1type, m_model->neq, 1);
    out["eq_obj2type"] = toNdarray2<int>(m_model->eq_obj2type, m_model->neq, 1);
    out["eq_obj1id"] = toNdarray2<int>(m_model->eq_obj1id, m_model->neq, 1);
    out["eq_obj2id"] = toNdarray2<int>(m_model->eq_obj2id, m_model->neq, 1);
    out["eq_size"] = toNdarray2<int>(m_model->eq_size, m_model->neq, 1);
    out["eq_ndata"] = toNdarray2<int>(m_model->eq_ndata, m_model->neq, 1);
    out["eq_isactive"] = toNdarray2<mjtByte>(m_model->eq_isactive, m_model->neq, 1);
    out["eq_compliance"] = toNdarray2<mjtNum>(m_model->eq_compliance, m_model->neq, 1);
    out["eq_timeconst"] = toNdarray2<mjtNum>(m_model->eq_timeconst, m_model->neq, 1);
    out["eq_data"] = toNdarray2<mjtNum>(m_model->eq_data, m_model->neq, 6);
    out["eq_user"] = toNdarray2<mjtNum>(m_model->eq_user, m_model->neq, m_model->nuser_eq);
    out["tendon_adr"] = toNdarray2<int>(m_model->tendon_adr, m_model->ntendon, 1);
    out["tendon_num"] = toNdarray2<int>(m_model->tendon_num, m_model->ntendon, 1);
    out["tendon_islimited"] = toNdarray2<mjtByte>(m_model->tendon_islimited, m_model->ntendon, 1);
    out["tendon_compliance"] = toNdarray2<mjtNum>(m_model->tendon_compliance, m_model->ntendon, 1);
    out["tendon_timeconst"] = toNdarray2<mjtNum>(m_model->tendon_timeconst, m_model->ntendon, 1);
    out["tendon_range"] = toNdarray2<mjtNum>(m_model->tendon_range, m_model->ntendon, 2);
    out["tendon_mindist"] = toNdarray2<mjtNum>(m_model->tendon_mindist, m_model->ntendon, 1);
    out["tendon_stiffness"] = toNdarray2<mjtNum>(m_model->tendon_stiffness, m_model->ntendon, 1);
    out["tendon_damping"] = toNdarray2<mjtNum>(m_model->tendon_damping, m_model->ntendon, 1);
    out["tendon_lengthspring"] = toNdarray2<mjtNum>(m_model->tendon_lengthspring, m_model->ntendon, 1);
    out["tendon_invweight0"] = toNdarray2<mjtNum>(m_model->tendon_invweight0, m_model->ntendon, 1);
    out["tendon_user"] = toNdarray2<mjtNum>(m_model->tendon_user, m_model->ntendon, m_model->nuser_tendon);
    out["wrap_type"] = toNdarray2<int>(m_model->wrap_type, m_model->nwrap, 1);
    out["wrap_objid"] = toNdarray2<int>(m_model->wrap_objid, m_model->nwrap, 1);
    out["wrap_prm"] = toNdarray2<mjtNum>(m_model->wrap_prm, m_model->nwrap, 1);
    out["actuator_dyntype"] = toNdarray2<int>(m_model->actuator_dyntype, m_model->nu, 1);
    out["actuator_trntype"] = toNdarray2<int>(m_model->actuator_trntype, m_model->nu, 1);
    out["actuator_gaintype"] = toNdarray2<int>(m_model->actuator_gaintype, m_model->nu, 1);
    out["actuator_biastype"] = toNdarray2<int>(m_model->actuator_biastype, m_model->nu, 1);
    out["actuator_trnid"] = toNdarray2<int>(m_model->actuator_trnid, m_model->nu, 2);
    out["actuator_isctrllimited"] = toNdarray2<mjtByte>(m_model->actuator_isctrllimited, m_model->nu, 1);
    out["actuator_isforcelimited"] = toNdarray2<mjtByte>(m_model->actuator_isforcelimited, m_model->nu, 1);
    out["actuator_dynprm"] = toNdarray2<mjtNum>(m_model->actuator_dynprm, m_model->nu, mjNDYN);
    out["actuator_trnprm"] = toNdarray2<mjtNum>(m_model->actuator_trnprm, m_model->nu, mjNTRN);
    out["actuator_gainprm"] = toNdarray2<mjtNum>(m_model->actuator_gainprm, m_model->nu, mjNGAIN);
    out["actuator_biasprm"] = toNdarray2<mjtNum>(m_model->actuator_biasprm, m_model->nu, mjNBIAS);
    out["actuator_ctrlrange"] = toNdarray2<mjtNum>(m_model->actuator_ctrlrange, m_model->nu, 2);
    out["actuator_forcerange"] = toNdarray2<mjtNum>(m_model->actuator_forcerange, m_model->nu, 2);
    out["actuator_invweight0"] = toNdarray2<mjtNum>(m_model->actuator_invweight0, m_model->nu, 1);
    out["actuator_length0"] = toNdarray2<mjtNum>(m_model->actuator_length0, m_model->nu, 1);
    out["actuator_lengthrange"] = toNdarray2<mjtNum>(m_model->actuator_lengthrange, m_model->nu, 2);
    out["actuator_user"] = toNdarray2<mjtNum>(m_model->actuator_user, m_model->nu, m_model->nuser_actuator);
    out["numeric_adr"] = toNdarray2<int>(m_model->numeric_adr, m_model->nnumeric, 1);
    out["numeric_size"] = toNdarray2<int>(m_model->numeric_size, m_model->nnumeric, 1);
    out["numeric_data"] = toNdarray2<mjtNum>(m_model->numeric_data, m_model->nnumericdata, 1);
    out["text_adr"] = toNdarray2<int>(m_model->text_adr, m_model->ntext, 1);
    out["text_data"] = toNdarray2<char>(m_model->text_data, m_model->ntextdata, 1);
    out["name_bodyadr"] = toNdarray2<int>(m_model->name_bodyadr, m_model->nbody, 1);
    out["name_jntadr"] = toNdarray2<int>(m_model->name_jntadr, m_model->njnt, 1);
    out["name_geomadr"] = toNdarray2<int>(m_model->name_geomadr, m_model->ngeom, 1);
    out["name_siteadr"] = toNdarray2<int>(m_model->name_siteadr, m_model->nsite, 1);
    out["name_meshadr"] = toNdarray2<int>(m_model->name_meshadr, m_model->nmesh, 1);
    out["name_hfieldadr"] = toNdarray2<int>(m_model->name_hfieldadr, m_model->nhfield, 1);
    out["name_eqadr"] = toNdarray2<int>(m_model->name_eqadr, m_model->neq, 1);
    out["name_tendonadr"] = toNdarray2<int>(m_model->name_tendonadr, m_model->ntendon, 1);
    out["name_actuatoradr"] = toNdarray2<int>(m_model->name_actuatoradr, m_model->nu, 1);
    out["name_numericadr"] = toNdarray2<int>(m_model->name_numericadr, m_model->nnumeric, 1);
    out["name_textadr"] = toNdarray2<int>(m_model->name_textadr, m_model->ntext, 1);
    out["names"] = toNdarray2<char>(m_model->names, m_model->nnames, 1);
    out["key_qpos"] = toNdarray2<mjtNum>(m_model->key_qpos, mjNKEY, m_model->nq);
    out["key_qvel"] = toNdarray2<mjtNum>(m_model->key_qvel, mjNKEY, m_model->nv);
    out["key_act"] = toNdarray2<mjtNum>(m_model->key_act, mjNKEY, m_model->na);
    out["cam_objtype"] = toNdarray2<int>(m_model->cam_objtype, m_model->ncam, 1);
    out["cam_objid"] = toNdarray2<int>(m_model->cam_objid, m_model->ncam, 1);
    out["cam_resolution"] = toNdarray2<int>(m_model->cam_resolution, m_model->ncam, 2);
    out["cam_fov"] = toNdarray2<mjtNum>(m_model->cam_fov, m_model->ncam, 2);
    out["cam_ipd"] = toNdarray2<mjtNum>(m_model->cam_ipd, m_model->ncam, 1);