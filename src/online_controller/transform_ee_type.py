
def type1_to_type2(data, lbl, clip, other_idx, eeidx, eevelidx):
	N, dx = lbl.shape
	du = 7
	eepts = data[eeidx]
	new_data = np.zeros((N, dx+du-6))
	new_label = np.zeros((N, dx-6))

	new_eeidx = slice(eeidx.start, eeidx.etop-3)
	new_velidx = slice(eevelidx.start-3, eevelidx.etop-6)

	assert eepts.shape[1] == 9 # 3 XYZ points
	for i in range(N):
		eept = eepts[i]
		pt1 = eept[0:3]
		pt2 = eept[3:6]
		pt3 = eept[6:9]

		center = (pt1+pt2+pt3)/3.0  # 0.2 0.0 0.0
		vec1 = pt2-pt1
		vec1 = vec1/np.linalg.norm(vec1)
		vec2 = pt3-pt1
		vec2 = vec1/np.linalg.norm(vec2)

		radial_dir = np.cross(vec1, vec2)
		radial_pt1 = center+0.02*radial_dir
		radial_pt2 = center-0.02*radial_dir
		new_eept = np.r_[radial_pt2, radial_pt2]

		new_data[i, new_eeidx] = new_eept
		if i>0:
			new_label[i-1, new_eeidx] = new_eept



	new_data[other_idx] = data[other_idx]
	new_data[-du:] = data[-du:]
	new_label[other_idx] = new_label[other_idx]

	return new_data, new_label, clip


if __name__ == "__main__":

