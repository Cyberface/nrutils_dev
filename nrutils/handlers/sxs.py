
#
from nrutils.core.basics import smart_object, parent
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sign
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # the folder is valid if there is l=m=2 mode data in the following dirs
    status = exist( run_dir + '/OutermostExtraction/rMPsi4_Y_l2_m2.asc' )\
             and exist( run_dir + '/Extrapolated_N2/rMPsi4_Y_l2_m2.asc' )\
             and exist( run_dir + '/Extrapolated_N3/rMPsi4_Y_l2_m2.asc' )\
             and exist( run_dir + '/Extrapolated_N4/rMPsi4_Y_l2_m2.asc' );

    #
    return status

#
def learn_metadata( metadata_file_location ):

    #
    from numpy import array

    #
    raw_metadata = smart_object( metadata_file_location )
    # shortand
    y = raw_metadata

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
    x.m1 = y.initial_mass1
    x.m2 = y.initial_mass2

    #
    J = array(y.initial_ADM_angular_momentum)
    S1 = array(y.initial_spin1); S2 = array(y.initial_spin2)
    S = S1 + S2
    L = J-S
    P = y.initial_ADM_linear_momentum

    # Prepare to deduce initial linear momenta
    R1 = array(y.initial_position1);
    R2 = array(y.initial_position2); rr = R2-R1
    R = array(   [  [0,rr[2],-rr[1]],  [-rr[2],0,rr[0]],  [rr[1],-rr[0],0]  ]   )
    H = L - cross( y.initial_position2, P )

    #
    rmap = abs( asum(R,axis=0) ) > 1e-6;
    k = next(k for k in range(len(rmap)) if rmap[k])

    P1 = zeros( P.shape )
    P1[k:] = dot( inv(R[k:,k:]), H[k:] )
    P2 = array(y.initial_ADM_linear_momentum) - P1

    #
    x.note = 'The SXS metadata give only total initial linear and angular momenta. In this code, the momenta of each BH has been deduced from basic linear algebra. However, this appraoch does not constrain the X COMPONENT of the total algular momentum, resulting in disagreement between the meta data value, and the value resulting from the appropriate sum. Use the metadata value.'

    #
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    #
    B = L1 + L2
    if norm( L[k:] - B[k:] ) > 1e-6 :
        print '>> norm( L[k:] - B[k:] ) = %f > 1e-6' % (norm( L[k:] - B[k:] ))
        msg = '>> Inconsistent handling of initial momenta. Please scrutinize.'
        raise ValueError(msg)

    #
    x.madm = y.initial_ADM_energy

    #
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    #
    x.b = float( y.initial_separation )
    if abs( x.b - norm(R1-R2) ) > 1e-6:
        msg = '(!!) Inconsistent assignment of initial separation.'
        raise ValueError(msg)

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    #
    x.valid = True

    #
    x.mf = y.remnant_mass

    #
    x.Sf = y.remnant_spin
    x.Xf = x.Sf/(x.mf*x.mf)
    x.xf = sign(x.Sf[-1])*norm(x.Sf)/(x.mf**2)

    # NOTE that I'm not sure about the units of all of these quantities. In particular, there are cases where the initial mass is not scaled to 1. This is inconsistent with nrutils' conventions and we wish to correct that here.
    # NOTE that the order of the lines below matters significantly.

    M = x.m1+x.m2

    _m1 = x.m1/M
    _m2 = x.m2/M
    x.S1 = _m1*_m1*x.S1/(x.m1*x.m1)
    x.S2 = _m2*_m2*x.S2/(x.m2*x.m2)
    x.m1,x.m2 = _m1,_m2

    _mf = y.remnant_mass/M
    x.Sf = _mf*_mf*x.Sf/(x.mf*x.mf)
    x.mf = _mf

    # QUESTION: Should the momenta and distances also be rescaled?

    #
    return standard_metadata, raw_metadata


# Given an extraction parameter, return an extraction radius
def extraction_map( this,
                    extraction_parameter,  ):

    #
    from numpy import inf

    # NOTE that the current set of sxs data is extrapolated to infinity -- this may change if finite radius data is downloaded
    extraction_radius = inf

    #
    return extraction_radius
