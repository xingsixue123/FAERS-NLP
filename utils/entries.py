"""
Entries Recorded:

general:
    safetyreportversion: int
    safetyreportid: str
    primarysourcecountry: str, country code label
    occurcountry: str, country code label
    reporttype: int, label
    serious: int, label
        serious
        seriousnesscongenitalanomali
        seriousnessdeath
        seriousnessdisabling
        seriousnesshospitalization
        seriousnesslifethreatening
        seriousnessother
    receivedate: date str
    transmissiondate: date str
    companynumb: str

sender:
    sendertype: int, label
    senderorganization: str
    
patient:
    patientonsetage: int
    patientonsetageunit: int, label
    patientsex: int, label
    
    reaction(multi):
        reactionmeddrapt: str
        reactionoutcome: int, label
    
    drug(multi):
        drugcharacterization: int, label
        medicinalproduct: str
        drugdosagetext: str
        drugindicational
        drugindication: str
        actiondrug: int, label
        drugadditional: 
        
        activesubstance(multi):
            activesubstancename: str


"""

