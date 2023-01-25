def from_swc_array(data):
    from ngauge import Neuron, TracingPoint
    out = Neuron()
    metadata = ""
    soma_pts = []
    soma_ids = set()
    branches = []
    soma_lookup = {}
    branch_lookup = {}

    for line in data:
        pt_num = int(line[0])
        pt_type = int(line[1])
        pt_x = float(line[2])
        pt_y = float(line[3])
        pt_z = float(line[4])
        pt_r = float(line[5])
        pt_parent = int(line[6])

        if pt_type == 1:  # soma
            toadd = (pt_x, pt_y, pt_z, pt_r)
            soma_pts.append(toadd)
            soma_ids.add(pt_num)
            soma_lookup[pt_num] = toadd
        else:
            child = TracingPoint(pt_x, pt_y, pt_z, pt_r, pt_type, fid=pt_num)
            if pt_parent in soma_ids:  # root of branch
                parent = soma_lookup[pt_parent]
                parent = TracingPoint(
                    x=parent[0],
                    y=parent[1],
                    z=parent[2],
                    r=parent[3],
                    t=1,
                    fid=pt_parent,
                )
                child.parent = parent
                parent.add_child(child)

                branch_lookup[pt_parent] = parent
                branch_lookup[pt_num] = child

                branches.append(parent)
            elif pt_parent == -1:  # root of branch that doesn't connect to soma
                child.parent = None
                branch_lookup[pt_num] = child
                branches.append(
                    child
                )  # add branch for child since this is complicated
            else:  # failed lookup
                if pt_parent not in branch_lookup:
                    raise ValueError("Parent id %d not present" % (pt_parent))
                branch_lookup[pt_parent].add_child(child)
                child.parent = branch_lookup[pt_parent]
                branch_lookup[pt_num] = child

    out.add_soma_points(soma_pts)
    for i in branches:
        out.add_branch(i)
    out.metadata = metadata
    return out