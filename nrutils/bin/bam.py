
#
from nrutils.core.basics import smart_object,parent,blue,smart_load, textul, red
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join,basename
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # The folder is valid if there is l=m=2 mode data in the following dirs
    status = len( ls( run_dir + '/Psi4ModeDecomp/psi3col*l2.m2.gz' ) ) > 0

    # ignore directories with certain tags in filename
    ignore_tags = ['backup','old']
    for tag in ignore_tags:
        status = status and not ( tag in run_dir )

    #
    a = basename(metadata_file_location).split(config.metadata_id)[0]
    b = parent(metadata_file_location)
    status = status and (  a in b  )
    # this checks that we get the correct .bbh file.
    # next we perform a check that the metadata contains numbers and not
    # mathematica errors such as 'BBHReduce`Private`content$28444'

    # Here is status is True then we have found the correct .bbh file.
    # This .bbh file should then have the metadata that we want i.e. masses etc.
    # Here we can then check that the values in the fields are in fact floats.
    # If we don't perform this check then errors can occur for instance
    # when we multiple the puncutre data by the masses to get momentum
    if status is True:
        raw_metadata = smart_object( metadata_file_location )
        # shortand
        y = raw_metadata

        # Check that inital mass1 is float
        if not isinstance( y.mass1 , float ) and not isinstance( y.mass2 , float ):
            msg = 'This is a valid .bbh file however mass1 and mass2 are not floats!' + \
            ' This is therefore invalid and is being skipped. Setting status to False.'
            print( red(msg) )
            status = False
    else:
        pass

    #
    return status

#
def learn_metadata( metadata_file_location ):

    #
    raw_metadata = smart_object( metadata_file_location )
    # shortand
    y = raw_metadata

    # check that the metadata contain the required attributes and
    # that they are floats. If not then set them to -1.
    required_attrs = [ 'mass1',
                       'mass2',
                       'initial_bh_momentum1x',
                       'initial_bh_momentum1y',
                       'initial_bh_momentum1z',
                       'initial_bh_momentum2x',
                       'initial_bh_momentum2y',
                       'initial_bh_momentum2z',
                       'initial_bh_spin1x',
                       'initial_bh_spin1y',
                       'initial_bh_spin1z',
                       'initial_bh_spin2x',
                       'initial_bh_spin2y',
                       'initial_bh_spin2z',
                       'after_junkradiation_spin1x',
                       'after_junkradiation_spin1y',
                       'after_junkradiation_spin1z',
                       'after_junkradiation_spin2x',
                       'after_junkradiation_spin2y',
                       'after_junkradiation_spin2z',
                       'initial_ADM_energy'
    ]

    for attr in required_attrs:
        if isinstance( getattr( y, attr ), ( float, int) ) is True:
            pass
        else:
            print( red( "{0} is not a float or int, setting value to -1".format( attr ) ) )
            setattr( y, attr, -1. )





    # # Useful for debuggin -- show what's in y
    # y.show()

    #
    standard_metadata = smart_object()
    # shorthand
    x = standard_metadata

    # Creation date of metadata file
    x.date_number = getctime(  metadata_file_location  )

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate derivative quantities  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Masses
    x.m1 = y.mass1
    x.m2 = y.mass2

    #
    P1 = array( [ y.initial_bh_momentum1x, y.initial_bh_momentum1y, y.initial_bh_momentum1z ] )
    P2 = array( [ y.initial_bh_momentum2x, y.initial_bh_momentum2y, y.initial_bh_momentum2z ] )

    #
    # This will break if there is after junkradiation spin estimates are empty.
    # First try and assign after junk spins. If resulting arrays are emtpy then use initial spins.
    S1 = array( [ y.after_junkradiation_spin1x, y.after_junkradiation_spin1y, y.after_junkradiation_spin1z ] )
    S2 = array( [ y.after_junkradiation_spin2x, y.after_junkradiation_spin2y, y.after_junkradiation_spin2z ] )
    if S1.size or S2.size == 0:
        S1 = array( [ y.initial_bh_spin1x, y.initial_bh_spin1y, y.initial_bh_spin1z ] )
        S2 = array( [ y.initial_bh_spin2x, y.initial_bh_spin2y, y.initial_bh_spin2z ] )

    # find puncture data locations
    puncture_data_1_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate1*' )[0]
    puncture_data_2_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate2*' )[0]

    # load puncture data
    puncture_data_1,_ = smart_load( puncture_data_1_location )
    puncture_data_2,_ = smart_load( puncture_data_2_location )

    after_junkradiation_time = 0 # y.after_junkradiation_time
    # Need two because the length of the puncture data could be different
    after_junkradiation_mask_1 = puncture_data_1[:,-1] > after_junkradiation_time
    after_junkradiation_mask_2 = puncture_data_2[:,-1] > after_junkradiation_time

    puncture_data_1 = puncture_data_1[ after_junkradiation_mask_1, : ]
    puncture_data_2 = puncture_data_2[ after_junkradiation_mask_2, : ]

    R1 = array( [  puncture_data_1[0,0],puncture_data_1[0,1],puncture_data_1[0,2],  ] )
    R2 = array( [  puncture_data_2[0,0],puncture_data_2[0,1],puncture_data_2[0,2],  ] )

    # print x.m1
    # print puncture_data_1

    P1 = x.m1 * array( [  puncture_data_1[0,3],puncture_data_1[0,4],puncture_data_1[0,5],  ] )
    P2 = x.m2 * array( [  puncture_data_2[0,3],puncture_data_2[0,4],puncture_data_2[0,5],  ] )

    # #
    # R1_old = array( [ y.initial_bh_position1x, y.initial_bh_position1y, y.initial_bh_position1z ] )
    # R2_old = array( [ y.initial_bh_position2x, y.initial_bh_position2y, y.initial_bh_position2z ] )
    #
    # print '>> old: %s' % R1_old
    # print '>> new: %s' % R1

    #
    x.note = ''

    #
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    #
    x.madm = y.initial_ADM_energy

    #
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    #
    x.b = float( y.initial_separation )
    if abs( x.b - norm(R1-R2) ) > 1e-4:
        msg = '(!!) Inconsistent assignment of initial separation: \n\t\tx = %f\n\t\tdR=%f' % (x.b,norm(R1-R2))
        print red(textul(msg))
        print red(textul('PASSING anyway'))
        pass
        # raise ValueError(msg)

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    #
    x.valid = True

    # Load irriducible mass data
    irr_mass_file_list = ls(parent(metadata_file_location)+'hmass_2*gz')
    if len(irr_mass_file_list)>0:
        irr_mass_file = irr_mass_file_list[0]
        irr_mass_data,mass_status = smart_load(irr_mass_file)
    else:
        mass_status = False
    # Load spin data
    spin_file_list = ls(parent(metadata_file_location)+'hspin_2*gz')
    if len(spin_file_list)>0:
        spin_file = spin_file_list[0]
        spin_data,spin_status = smart_load(spin_file)
    else:
        spin_status = False
    # Estimate final mass and spin
    if mass_status and spin_status:
        try:
            Sf = spin_data[-1,1:]
            irrMf = irr_mass_data[-1,1]
        except IndexError:
            Sf = spin_data[1:]
            irrMf = irr_mass_data[1]
        x.mf = sqrt( irrMf**2 + norm(Sf/irrMf)**2 )
        #
        x.Sf = Sf
        x.xf = norm(x.Sf)/(x.mf*x.mf)
    else:
        x.Sf = array([0.0,0.0,0.0])
        x.mf = 0.0
        x.xf = array([0.0,0.0,0.0])


    # True if ectraction parameter is extraction radius
    x.extraction_parameter_is_radius = False

    #
    return standard_metadata, raw_metadata
