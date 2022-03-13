import yaml

import argparse

parser = argparse.ArgumentParser(
    prog='python3 simulator_benchmark.py',
    description='Run AI Simulator with 8 models benchmark',
    )

parser.add_argument('-n', '--num_server', dest='num_server', default=2,
                    help='number of servers')
parser.add_argument('-b', '--bandwidth', dest='bandwidth', default='100Gibps',
                    help='bandwidth of network')
parser.add_argument('-l', '--latency', dest='latency', default='1us',
                    help='latency of network')

def gen_link_dict(dest, link_type, rate, propagation_latency, scheduler):
    link_dict = {'dest': dest,
                'type': link_type,
                'rate':rate,
                'propagation_latency': propagation_latency,
                'scheduler': scheduler}
    return link_dict

def gen_cpu_dict(switch_name):
    attr_dict = {}
    attr_dict['properties'] = {'average_performance' : '192Gibps'}
    link = gen_link_dict(switch_name, 'PCIE', '192Gibps', '1us', 'FIFO')
    attr_dict['links'] = [link]
    return attr_dict

def gen_8MI100_IB_pool(num_server, network_bd='100Gibps', network_lat='1us'):
    '''
    Each server has 2 CPU and 8 MI100, 4 MI100 under same CPU are connected via a switch.
    Network is connected by a virtual switch and a fat-tree topology.
    @param num_server: the number of servers.
    @network_bd: The bandwidth of IB network link.
    @network_lat: The latency of IB network link. 
    '''
    server_dict = {}
    switch_dict = {}
    resource_dict = {'Server' : server_dict, 'Switch' : switch_dict}

    # Generate server
    for server_id in range(num_server):
        hostname = 'hostname' + str(server_id)
        cpu_dict = {}
        gpu_dict = {}
        # gen 2 cpu
        for cpu_id in range(2):
            switch_name = '/switch/hostname%d/cpuswitch%d/' % (server_id, cpu_id)
            cpu_dict[cpu_id] = gen_cpu_dict(switch_name) 
        # gen 0-3 gpu
        for gpu_id in range(0, 4):
            new_gpu = {'properties':{'average_performance':'12Tibps'}}
            gpu_links = []
            # gpu 0-3 link to switch 0
            switch_name = '/switch/hostname%d/cpuswitch0/' % (server_id)
            switch_link = gen_link_dict(switch_name, 'PCIE', '192Gibps', '1us', 'FIFO')
            gpu_links.append(switch_link)
            # NVLink (type same as PCIE)
            for dst_gpu_id in range(0,4):
                if dst_gpu_id == gpu_id:
                    continue
                dst_gpu_name = '/server/hostname%d/GPU/%d/' % (server_id, dst_gpu_id)
                new_link = gen_link_dict(dst_gpu_name, 'PCIE', '300Gibps', '1us', 'FIFO')
                gpu_links.append(new_link)
            new_gpu['links'] = gpu_links
            gpu_dict[gpu_id] = new_gpu
        # gen 4-7 gpu
        for gpu_id in range(4, 8):
            new_gpu = {'properties':{'average_performance':'12Tibps'}}
            gpu_links = []
            # gpu 0-3 link to switch 0
            switch_name = '/switch/hostname%d/cpuswitch1/' % (server_id)
            switch_link = gen_link_dict(switch_name, 'PCIE', '192Gibps', '1us', 'FIFO')
            gpu_links.append(switch_link)
            # NVLink (type same as PCIE, 300gbps)
            for dst_gpu_id in range(4, 8):
                if dst_gpu_id == gpu_id:
                    continue
                dst_gpu_name = '/server/hostname%d/GPU/%d/' % (server_id, dst_gpu_id)
                new_link = gen_link_dict(dst_gpu_name, 'PCIE', '300Gibps', '1us', 'FIFO')
                gpu_links.append(new_link)
            new_gpu['links'] = gpu_links
            gpu_dict[gpu_id] = new_gpu

        divice_dict = {'CPU':cpu_dict, 'GPU':gpu_dict}
        server_dict[hostname] = divice_dict

        # Generate switch
        for server_id in range(num_server):
            hostname = 'hostname' + str(server_id)
            new_switch_dict = {}
            # Each cpu has a pcie switch
            for cpu_id in range(2):
                switch_name = '%s/cpuswitch%d' % (hostname, cpu_id)
                link_list = []
                # switch connect to CPU
                cpu_name = '/server/hostname%d/CPU/%d/' % (server_id, cpu_id)
                cpu_link = gen_link_dict(cpu_name, 'PCIE', '192Gibps', '1us', 'FIFO')
                link_list.append(cpu_link)
                # switch of cpu0 connect to switch of cpu1
                dst_switch_name = '/switch/hostname%d/cpuswitch%d/' % (server_id, (cpu_id+1)%2) 
                switch_link = gen_link_dict(dst_switch_name, 'PCIE', '192Gibps', '1us', 'FIFO')
                link_list.append(switch_link)
                # switch connect to GPU
                for local_gpu_id in range(4):
                    # Each CPU has 4 gpu under it
                    gpu_id =  local_gpu_id + 4*cpu_id
                    gpu_name = '/server/hostname%d/GPU/%d/' % (server_id, gpu_id)
                    new_gpu_link = gen_link_dict(gpu_name, 'PCIE', '192Gibps', '1us', 'FIFO')
                    link_list.append(new_gpu_link)
                
                # network connect to other switch
                if cpu_id == 0:
                    # Network port is under cpu0
                    for dst_server_id in range(num_server):
                        if dst_server_id == server_id:
                            continue
                        dst_switch_name = '/switch/hostname%d/cpuswitch0/' % (dst_server_id)
                        switch_link = gen_link_dict(dst_switch_name, 'RDMA', network_bd, network_lat, 'FIFO')
                        link_list.append(switch_link)

            # This part because in plan_gen, all switches are listed together, not server-switch structure.
                switch_dict[switch_name] = {'links':link_list}
                #new_switch_dict[switch_name] = {'links':link_list}
            #switch_dict[switch_name] = new_switch_dict
    return resource_dict

if __name__ == '__main__':
    args = parser.parse_args()

    resource_dict = gen_8MI100_IB_pool(num_server=int(args.num_server), 
                        network_bd=args.bandwidth, 
                        network_lat=args.latency)
    with open('test.yaml','w') as f:
        f.write(yaml.dump(resource_dict, indent=4))
    print('Generate done.')