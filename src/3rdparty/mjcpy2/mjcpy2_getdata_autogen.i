    out["nstack"] = m_data->nstack;
    out["nbuffer"] = m_data->nbuffer;
    out["pstack"] = m_data->pstack;
    out["maxstackuse"] = m_data->maxstackuse;
    out["ne"] = m_data->ne;
    out["nf"] = m_data->nf;
    out["nefc"] = m_data->nefc;
    out["ncon"] = m_data->ncon;
    out["nwarning"] = toNdarray1<int>(m_data->nwarning, mjNWARNING);
    out["timer_duration"] = toNdarray1<mjtNum>(m_data->timer_duration, mjNTIMER);
    out["timer_ncall"] = toNdarray1<mjtNum>(m_data->timer_ncall, mjNTIMER);
    out["mocaptime"] = toNdarray1<mjtNum>(m_data->mocaptime, 3);
    out["time"] = m_data->time;
    out["energy"] = toNdarray1<mjtNum>(m_data->energy, 2);
    out["solverstat"] = toNdarray1<mjtNum>(m_data->solverstat, 4);
    out["qpos"] = toNdarray2<mjtNum>(m_data->qpos, m_model->nq, 1);
    out["qvel"] = toNdarray2<mjtNum>(m_data->qvel, m_model->nv, 1);
    out["act"] = toNdarray2<mjtNum>(m_data->act, m_model->na, 1);
    out["ctrl"] = toNdarray2<mjtNum>(m_data->ctrl, m_model->nu, 1);
    out["qfrc_applied"] = toNdarray2<mjtNum>(m_data->qfrc_applied, m_model->nv, 1);
    out["xfrc_applied"] = toNdarray2<mjtNum>(m_data->xfrc_applied, m_model->nbody, 6);
    out["qacc"] = toNdarray2<mjtNum>(m_data->qacc, m_model->nv, 1);
    out["act_dot"] = toNdarray2<mjtNum>(m_data->act_dot, m_model->na, 1);
    out["mocap_pos"] = toNdarray2<mjtNum>(m_data->mocap_pos, m_model->nmocap, 3);
    out["mocap_quat"] = toNdarray2<mjtNum>(m_data->mocap_quat, m_model->nmocap, 4);
    out["userdata"] = toNdarray2<mjtNum>(m_data->userdata, m_model->nuserdata, 1);
    out["sensordata"] = toNdarray2<mjtNum>(m_data->sensordata, m_model->nsensordata, 1);
    out["xpos"] = toNdarray2<mjtNum>(m_data->xpos, m_model->nbody, 3);
    out["xquat"] = toNdarray2<mjtNum>(m_data->xquat, m_model->nbody, 4);
    out["xmat"] = toNdarray2<mjtNum>(m_data->xmat, m_model->nbody, 9);
    out["xipos"] = toNdarray2<mjtNum>(m_data->xipos, m_model->nbody, 3);
    out["ximat"] = toNdarray2<mjtNum>(m_data->ximat, m_model->nbody, 9);
    out["xanchor"] = toNdarray2<mjtNum>(m_data->xanchor, m_model->njnt, 3);
    out["xaxis"] = toNdarray2<mjtNum>(m_data->xaxis, m_model->njnt, 3);
    out["geom_xpos"] = toNdarray2<mjtNum>(m_data->geom_xpos, m_model->ngeom, 3);
    out["geom_xmat"] = toNdarray2<mjtNum>(m_data->geom_xmat, m_model->ngeom, 9);
    out["site_xpos"] = toNdarray2<mjtNum>(m_data->site_xpos, m_model->nsite, 3);
    out["site_xmat"] = toNdarray2<mjtNum>(m_data->site_xmat, m_model->nsite, 9);
    out["cam_xpos"] = toNdarray2<mjtNum>(m_data->cam_xpos, m_model->ncam, 3);
    out["cam_xmat"] = toNdarray2<mjtNum>(m_data->cam_xmat, m_model->ncam, 9);
    out["light_xpos"] = toNdarray2<mjtNum>(m_data->light_xpos, m_model->nlight, 3);
    out["light_xdir"] = toNdarray2<mjtNum>(m_data->light_xdir, m_model->nlight, 3);
    out["com_subtree"] = toNdarray2<mjtNum>(m_data->com_subtree, m_model->nbody, 3);
    out["cdof"] = toNdarray2<mjtNum>(m_data->cdof, m_model->nv, 6);
    out["cinert"] = toNdarray2<mjtNum>(m_data->cinert, m_model->nbody, 10);
    out["ten_wrapadr"] = toNdarray2<int>(m_data->ten_wrapadr, m_model->ntendon, 1);
    out["ten_wrapnum"] = toNdarray2<int>(m_data->ten_wrapnum, m_model->ntendon, 1);
    out["ten_length"] = toNdarray2<mjtNum>(m_data->ten_length, m_model->ntendon, 1);
    out["ten_moment"] = toNdarray2<mjtNum>(m_data->ten_moment, m_model->ntendon, m_model->nv);
    out["wrap_obj"] = toNdarray2<int>(m_data->wrap_obj, m_model->nwrap*2, 1);
    out["wrap_xpos"] = toNdarray2<mjtNum>(m_data->wrap_xpos, m_model->nwrap*2, 3);
    out["actuator_length"] = toNdarray2<mjtNum>(m_data->actuator_length, m_model->nu, 1);
    out["actuator_moment"] = toNdarray2<mjtNum>(m_data->actuator_moment, m_model->nu, m_model->nv);
    out["crb"] = toNdarray2<mjtNum>(m_data->crb, m_model->nbody, 10);
    out["qM"] = toNdarray2<mjtNum>(m_data->qM, m_model->nM, 1);
    out["qLD"] = toNdarray2<mjtNum>(m_data->qLD, m_model->nM, 1);
    out["qLDiagInv"] = toNdarray2<mjtNum>(m_data->qLDiagInv, m_model->nv, 1);
    out["qLDiagSqrtInv"] = toNdarray2<mjtNum>(m_data->qLDiagSqrtInv, m_model->nv, 1);
    out["efc_type"] = toNdarray2<int>(m_data->efc_type, m_model->njmax, 1);
    out["efc_id"] = toNdarray2<int>(m_data->efc_id, m_model->njmax, 1);
    out["efc_signature"] = toNdarray2<int>(m_data->efc_signature, m_model->njmax, 1);
    out["efc_rownnz"] = toNdarray2<int>(m_data->efc_rownnz, m_model->njmax, 1);
    out["efc_rowadr"] = toNdarray2<int>(m_data->efc_rowadr, m_model->njmax, 1);
    out["efc_colind"] = toNdarray2<int>(m_data->efc_colind, m_model->njmax, m_model->nv);
    out["efc_rownnz_T"] = toNdarray2<int>(m_data->efc_rownnz_T, m_model->nv, 1);
    out["efc_rowadr_T"] = toNdarray2<int>(m_data->efc_rowadr_T, m_model->nv, 1);
    out["efc_colind_T"] = toNdarray2<int>(m_data->efc_colind_T, m_model->nv, m_model->njmax);
    out["efc_solref"] = toNdarray2<mjtNum>(m_data->efc_solref, m_model->njmax, mjNREF);
    out["efc_solimp"] = toNdarray2<mjtNum>(m_data->efc_solimp, m_model->njmax, mjNIMP);
    out["efc_margin"] = toNdarray2<mjtNum>(m_data->efc_margin, m_model->njmax, 1);
    out["efc_frictionloss"] = toNdarray2<mjtNum>(m_data->efc_frictionloss, m_model->njmax, 1);
    out["efc_pos"] = toNdarray2<mjtNum>(m_data->efc_pos, m_model->njmax, 1);
    out["efc_J"] = toNdarray2<mjtNum>(m_data->efc_J, m_model->njmax, m_model->nv);
    out["efc_J_T"] = toNdarray2<mjtNum>(m_data->efc_J_T, m_model->nv, m_model->njmax);
    out["efc_diagApprox"] = toNdarray2<mjtNum>(m_data->efc_diagApprox, m_model->njmax, 1);
    out["efc_R"] = toNdarray2<mjtNum>(m_data->efc_R, m_model->njmax, 1);
    out["efc_AR"] = toNdarray2<mjtNum>(m_data->efc_AR, m_model->njmax, m_model->njmax);
    out["e_ARchol"] = toNdarray2<mjtNum>(m_data->e_ARchol, m_model->nemax, m_model->nemax);
    out["fc_e_rect"] = toNdarray2<mjtNum>(m_data->fc_e_rect, m_model->njmax, m_model->nemax);
    out["fc_AR"] = toNdarray2<mjtNum>(m_data->fc_AR, m_model->njmax, m_model->njmax);
    out["ten_velocity"] = toNdarray2<mjtNum>(m_data->ten_velocity, m_model->ntendon, 1);
    out["actuator_velocity"] = toNdarray2<mjtNum>(m_data->actuator_velocity, m_model->nu, 1);
    out["cvel"] = toNdarray2<mjtNum>(m_data->cvel, m_model->nbody, 6);
    out["cdof_dot"] = toNdarray2<mjtNum>(m_data->cdof_dot, m_model->nv, 6);
    out["qfrc_bias"] = toNdarray2<mjtNum>(m_data->qfrc_bias, m_model->nv, 1);
    out["qfrc_passive"] = toNdarray2<mjtNum>(m_data->qfrc_passive, m_model->nv, 1);
    out["efc_vel"] = toNdarray2<mjtNum>(m_data->efc_vel, m_model->njmax, 1);
    out["efc_aref"] = toNdarray2<mjtNum>(m_data->efc_aref, m_model->njmax, 1);
    out["actuator_force"] = toNdarray2<mjtNum>(m_data->actuator_force, m_model->nu, 1);
    out["qfrc_actuator"] = toNdarray2<mjtNum>(m_data->qfrc_actuator, m_model->nv, 1);
    out["qfrc_unc"] = toNdarray2<mjtNum>(m_data->qfrc_unc, m_model->nv, 1);
    out["qacc_unc"] = toNdarray2<mjtNum>(m_data->qacc_unc, m_model->nv, 1);
    out["efc_b"] = toNdarray2<mjtNum>(m_data->efc_b, m_model->njmax, 1);
    out["fc_b"] = toNdarray2<mjtNum>(m_data->fc_b, m_model->njmax, 1);
    out["efc_force"] = toNdarray2<mjtNum>(m_data->efc_force, m_model->njmax, 1);
    out["qfrc_constraint"] = toNdarray2<mjtNum>(m_data->qfrc_constraint, m_model->nv, 1);
    out["qfrc_inverse"] = toNdarray2<mjtNum>(m_data->qfrc_inverse, m_model->nv, 1);
    out["cacc"] = toNdarray2<mjtNum>(m_data->cacc, m_model->nbody, 6);
    out["cfrc_int"] = toNdarray2<mjtNum>(m_data->cfrc_int, m_model->nbody, 6);
    out["cfrc_ext"] = toNdarray2<mjtNum>(m_data->cfrc_ext, m_model->nbody, 6);
