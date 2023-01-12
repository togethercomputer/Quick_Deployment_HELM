/etc/singularity shell \
            --home $PWD:/srv \
            --pwd /srv \
            --bind /cvmfs \
            --scratch /var/tmp \
            --scratch /tmp \
            --contain --ipc --pid \
            /cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-ubuntu-20.04:latest/