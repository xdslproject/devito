# adapt the following line according your machine

# setup
NX=176
NY=176
NZ=176
ORDER=8

# tests all tile shapes where BLOCK_X*BLOCK_Y*BLOCK_Z <= 1024 and multiple of 32
for BLOCK_X in 2 4 8 16 32 64 128 256; do
    for BLOCK_Y in 2 4 8 16 32 64 128 256; do
        for BLOCK_Z in 2 4 8 16 32 64 128 256; do
            PRODUCT=$(($BLOCK_X*$BLOCK_Y*$BLOCK_Z))
            if [ $PRODUCT -le 1024 ] && [ $(($PRODUCT%32)) -eq 0 ]
            then
                echo "==========" $BLOCK_X $BLOCK_Y $BLOCK_Z
                export TILE_SIZE=$BLOCK_X,$BLOCK_Y,$BLOCK_Z
                DEVITO_PLATFORM=nvidiaX DEVITO_LANGUAGE=openacc DEVITO_ARCH=pgcc DEVITO_LOGGING=DEBUG python examples/seismic/acoustic/acoustic_example.py -so $ORDER -d $NX $NY $NZ
            fi
        done
    done
done
# acoustic_example.py is edited to save the results to disk in csv format (results.csv)
# each line contains: BLOCK_X,BLOCK_Y,BLOCK_Z,gflopss,gpointss,time,gflopss(section0),gpointss(section0),time(section0) 
