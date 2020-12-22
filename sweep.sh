# adapt the following line according your machine

# setup
NX=600
NY=600
NZ=600
ORDER=12

# tests all tile shapes where BLOCK_X*BLOCK_Y*BLOCK_Z <= 1024 and multiple of 32
for BLOCK_X in {16..64..8}; do
    for BLOCK_Y in {2..64..2}; do
        for BLOCK_Z in {2..64..2}; do
            PRODUCT=$(($BLOCK_X*$BLOCK_Y*$BLOCK_Z))
            if [ $PRODUCT -le 1024 ]
            then
                echo "==========" $BLOCK_X $BLOCK_Y $BLOCK_Z
                export TILE_SIZE=$BLOCK_X,$BLOCK_Y,$BLOCK_Z
                DEVITO_PLATFORM=nvidiaX DEVITO_LANGUAGE=openacc DEVITO_ARCH=pgcc DEVITO_LOGGING=DEBUG python examples/seismic/elastic/elastic_example.py -so $ORDER -d $NX $NY $NZ --tn 100
            fi
        done
    done
done
# acoustic_example.py is edited to save the results to disk in csv format (results.csv)
# each line contains: BLOCK_X,BLOCK_Y,BLOCK_Z,gflopss,gpointss,time,gflopss(section0),gpointss(section0),time(section0) 
