C Minimal test fixture derived from DSSAT/dssat-csm-os v4.8.0.24.
C Official source commit: caaa55c6bee21aa894b325b67dad7ccacb05295b.
C This excerpt is not a complete or buildable upstream source file.
      READ(CXCRD,'(F15.0)', IOSTAT=ERRNUM) XCRD
      IF(ERRNUM .NE. 0) THEN
         XCRD = -999.0
         MSG(1) = 'Error reading latitude from experimental file'
         MSG(2) = FILEX
         CALL WARNING(2, ERRKEY, MSG)
      ENDIF
      READ(CYCRD,'(F15.0)', IOSTAT=ERRNUM) YCRD
      IF(ERRNUM .NE. 0) THEN
         YCRD = -99.0
         MSG(1) = 'Error reading longitude from experimental file'
         MSG(2) = FILEX
         CALL WARNING(2, ERRKEY, MSG)
      ENDIF
      READ(CELEV,'(F9.0)', IOSTAT=ERRNUM)  ELEV
      IF(ERRNUM .NE. 0) THEN
        ELEV = -99.0
        MSG(1) = 'Error reading elevation from experimental file'
        MSG(2) = FILEX
        CALL WARNING(2, ERRKEY, MSG)
      ENDIF
      IF(YCRD .GE. -90.0 .AND. YCRD .LE. 90.0 .AND.
     &   XCRD .GE.-180.0 .AND. XCRD .LE. 180.0 .AND.
     &   LEN_TRIM(CYCRD).GT. 0.0 .AND. LEN_TRIM(CXCRD).GT.0.0
     &   .AND.
     &   (ABS(YCRD) .GT. 1.E-15 .OR. ABS(XCRD) .GT. 1.E-15))THEN
          CALL PUT('FIELD','CYCRD',CYCRD)
          CALL PUT('FIELD','CXCRD',CXCRD)
      ELSE
          CALL PUT('FIELD','CYCRD','            -99')
          CALL PUT('FIELD','CXCRD','            -99')
      ENDIF
