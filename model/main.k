$# LS-DYNA Keyword file created by LS-PrePost 4.2 (Beta) - 12Oct2014(09:00)
$# Created on May-07-2015 (10:59:09)
*KEYWORD
*CONTROL_SHELL
$#  wrpang     esort     irnxx    istupd    theory       bwc     miter      proj
 20.000000         0        -1         0         2         2         1         0
$# rotascl    intgrd    lamsht    cstyp6    tshell      
  1.000000         0         0         1         0
$# psstupd   sidt4tu     cntco    itsflg    irquad      
         0         0         0         0         2
$#  nfail1    nfail4   psnfail    keepcs     delfr   drcpsid    drcprm      
         0         0         0         0         0         0  1.000000
*CONTROL_TIMESTEP
$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
     0.000  0.600000         0     0.000     0.000         0         0         0
$#  dt2msf   dt2mslc     imscl    unused    unused     rmscl     

*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         6         6         3         1         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         1         0         1         1         1         2         1
$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout
         0         0  1.000000         0         0         0STRESS    STRESS
$#    dtdt    resplt      
         0         0
*PART
$#                                                                         title
spec-fr1
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         1         1         1         0         0         0         0         0
*SECTION_SHELL
$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp
         1        15  1.000000         2  1.000000     0.000         0         1
$#      t1        t2        t3        t4      nloc     marea      idof    edgset
     0.000     0.000     0.000     0.000     0.000     0.000     0.000         0
*BOUNDARY_SPC_SET
$#    nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz
         1         0         1         1         0         0         0         0
$#    nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz
         2         0         1         0         0         0         0         0
*BOUNDARY_PRESCRIBED_MOTION_SET
$#    nsid       dof       vad      lcid        sf       vid     death     birth
         2         2         0        99       1.0         01.00000E28       0.0
*INCLUDE
solid.k
*END
