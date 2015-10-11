with open('mjcpy2_getdata_autogen.i', 'rb') as f:
    in_lines = f.readlines()

with open('out.txt', 'wb') as f:
    for line in in_lines:
        quote_idxs = (line.index('"'), line.rindex('"'))
        field = line[(quote_idxs[0]+1):quote_idxs[1]]
        if 'array' in line:
            f.write('    _cadihk(d, "{0}", m_data->{0});\n'.format(field))
        else:
            f.write('    _csdihk(d, "{0}", m_data->{0});\n'.format(field))
