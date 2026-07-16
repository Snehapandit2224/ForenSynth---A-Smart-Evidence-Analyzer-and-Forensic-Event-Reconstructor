"""
ForenSynth-X+ Domains
Domain definitions, FIR templates, and context generation logic.

FIR descriptions are now template-aware — each template gets its own
pool of description strings that reflect the specific nature of the crime
(e.g. skimming vs. two-person robbery vs. insider theft).
"""

import random
from dataclasses import dataclass
from typing import Literal

Domain = Literal["ATM_Robbery", "Office_Theft", "Communication"]

DOMAINS: list[Domain] = ["ATM_Robbery", "Office_Theft", "Communication"]


@dataclass
class DomainSpec:
    name: Domain
    crime_types: list[str]
    locations: list[str]
    template_descriptions: dict[str, list[str]]   # template_name → description pool
    time_window_range: tuple[int, int]             # (min_duration_sec, max_duration_sec)


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "ATM_Robbery": DomainSpec(
        name="ATM_Robbery",
        crime_types=["ATM robbery", "cash machine theft", "ATM skimming with assault"],
        locations=[
            "MG Road ATM Booth",
            "Koramangala SBI ATM",
            "Indiranagar HDFC ATM",
            "Whitefield ATM Kiosk",
            "Jayanagar ATM Lobby",
        ],
        template_descriptions={
            "Entry_Action_Exit": [
                (
                    "A single individual was observed entering an ATM booth and conducting "
                    "a transaction under suspicious circumstances. {witness_count} bystander(s) "
                    "reported unusual behaviour before and after the incident."
                ),
                (
                    "Complainant reports being followed to an ATM by {suspect_count} unknown "
                    "individual(s) who subsequently used the machine. {witness_count} witness(es) "
                    "corroborate presence of the suspect near the kiosk."
                ),
                (
                    "CCTV footage shows {suspect_count} person(s) entering the ATM booth alone. "
                    "A subsequent check revealed unauthorised cash withdrawal. "
                    "{witness_count} bystander(s) observed the suspect at the scene."
                ),
                (
                    "Deponent states that {suspect_count} unidentified accused entered the ATM enclosure "
                    "and conducted a transaction without presenting valid credentials. "
                    "{witness_count} independent witness(es) were present at the time of the incident."
                ),
                (
                    "Bank security flagged an anomalous withdrawal at this ATM kiosk. "
                    "Review of surveillance footage confirms {suspect_count} individual(s) accessed "
                    "the terminal. {witness_count} person(s) in the vicinity provided eyewitness accounts."
                ),
                (
                    "Complainant reports an unauthorised transaction conducted at the ATM by "
                    "{suspect_count} accused person(s) during off-peak hours. "
                    "{witness_count} bystander(s) have corroborated the account."
                ),
                (
                    "A member of the public reported witnessing {suspect_count} individual(s) "
                    "behaving suspiciously at the ATM before departing hastily. "
                    "{witness_count} witness(es) have been identified and their statements recorded."
                ),
                (
                    "The complainant discovered an unauthorised debit from their account following "
                    "the attendance of {suspect_count} unknown person(s) at the ATM terminal. "
                    "{witness_count} witness(es) observed the accused at the location."
                ),
                (
                    "Bank records indicate a transaction initiated by {suspect_count} unregistered "
                    "individual(s) at this ATM. The incident was reported by {witness_count} "
                    "bystander(s) who noticed suspicious conduct near the machine."
                ),
                (
                    "Surveillance logs confirm entry by {suspect_count} person(s) into the ATM booth "
                    "during the reported time window. The accused departed prior to security arrival. "
                    "{witness_count} independent observer(s) corroborate the sequence of events."
                ),
                (
                    "The deponent alleges that {suspect_count} accused person(s) tampered with their "
                    "account at the ATM terminal. The incident occurred in the presence of "
                    "{witness_count} bystander(s) whose statements are on record."
                ),
                (
                    "A complaint has been lodged regarding the unlawful use of an ATM terminal by "
                    "{suspect_count} accused individual(s). {witness_count} eyewitness(es) have "
                    "confirmed the presence of the suspect(s) at the time and place of the incident."
                ),
            ],
            "Entry_Suspicious_Exit": [
                (
                    "An individual was seen loitering near an ATM for an extended period before "
                    "entering the booth. Witnesses reported the person appeared to tamper with the "
                    "card slot area before leaving hastily. {witness_count} witness(es) have provided statements."
                ),
                (
                    "Complaint received regarding a suspect who spent an unusual amount of time "
                    "at the ATM panel without completing a normal transaction. "
                    "{witness_count} bystander(s) noted suspicious handling of the machine."
                ),
                (
                    "ATM skimming device suspected. {suspect_count} individual(s) were observed "
                    "interfering with the card reader mechanism. The suspect fled the scene before "
                    "security personnel arrived. {witness_count} person(s) witnessed the incident."
                ),
                (
                    "Deponent reports that {suspect_count} accused person(s) were seen pacing near "
                    "the ATM kiosk for an extended duration before accessing the card reader. "
                    "{witness_count} bystander(s) noted the suspicious conduct and have filed statements."
                ),
                (
                    "A complaint has been filed alleging that the accused affixed a foreign device "
                    "to the ATM card slot before fleeing. {witness_count} eyewitness(es) observed "
                    "the suspect(s) at the machine during the relevant time window."
                ),
                (
                    "Bank technicians discovered signs of tampering on the ATM card reader consistent "
                    "with a skimming attempt. CCTV confirms {suspect_count} individual(s) accessed "
                    "the panel. {witness_count} member(s) of the public corroborate this account."
                ),
                (
                    "The complainant alleges that {suspect_count} accused individual(s) loitered "
                    "near the ATM for an abnormal duration and were observed manipulating the machine "
                    "before departing. {witness_count} independent witness(es) support this account."
                ),
                (
                    "Security personnel reported {suspect_count} person(s) exhibiting suspicious "
                    "behaviour near the ATM terminal, including extended dwell time and physical "
                    "contact with the card entry slot. {witness_count} bystander(s) provided statements."
                ),
                (
                    "A skimming device was recovered from the ATM card slot following a tip-off. "
                    "Footage confirms {suspect_count} suspect(s) had accessed the panel earlier. "
                    "{witness_count} witness(es) have confirmed their presence at the scene."
                ),
                (
                    "The deponent states that the ATM appeared to have been interfered with, "
                    "consistent with a card-cloning attempt by {suspect_count} unidentified accused. "
                    "{witness_count} person(s) in the vicinity at the relevant time have given accounts."
                ),
                (
                    "Complaint lodged after {suspect_count} individual(s) were seen inspecting "
                    "the ATM card reader without conducting any transaction, then departing abruptly "
                    "upon the arrival of another customer. {witness_count} witness(es) are on record."
                ),
                (
                    "Surveillance footage reviewed by the investigating officer shows {suspect_count} "
                    "person(s) interacting with the ATM card slot in a manner inconsistent with "
                    "normal use. {witness_count} independent observer(s) corroborate the account."
                ),
            ],
            "MultiActor_Entry_Action_Exit": [
                (
                    "Two individuals were seen entering an ATM booth in a coordinated manner. "
                    "One appeared to conduct a transaction while the other stood watch near the entrance. "
                    "{witness_count} bystander(s) reported the suspicious activity to local authorities."
                ),
                (
                    "{suspect_count} persons were observed acting in coordination near the ATM kiosk. "
                    "One entered the booth while the other remained outside as a lookout. "
                    "{witness_count} witness(es) corroborate the coordinated nature of the activity."
                ),
                (
                    "Complainant reports being coerced at an ATM by {suspect_count} unknown assailant(s) "
                    "who acted in concert. One suspect operated the machine while the other blocked the "
                    "entrance. {witness_count} independent witness(es) observed the incident."
                ),
                (
                    "Deponent states that {suspect_count} accused persons approached the ATM in tandem. "
                    "One accused conducted the transaction while the second monitored the surroundings "
                    "to deter intervention. {witness_count} bystander(s) have corroborated this account."
                ),
                (
                    "A coordinated ATM robbery was reported involving {suspect_count} accused individuals. "
                    "Division of roles was evident from CCTV footage, with one party operating the "
                    "terminal and another standing guard. {witness_count} eyewitness(es) are on record."
                ),
                (
                    "The complainant was confronted by {suspect_count} accused at the ATM kiosk. "
                    "The accused acted in a pre-planned manner, with one handling the machine and "
                    "another obstructing exit. {witness_count} witness(es) observed part of the incident."
                ),
                (
                    "Surveillance footage confirms the arrival of {suspect_count} individual(s) at "
                    "the ATM in a coordinated sequence. The accused executed a joint withdrawal before "
                    "departing together. {witness_count} person(s) present at the time have given statements."
                ),
                (
                    "Bank security raised an alert after {suspect_count} individuals approached the "
                    "ATM together, with one acting as a lookout while the other accessed the terminal. "
                    "{witness_count} bystander(s) independently confirmed the coordinated behaviour."
                ),
                (
                    "A complaint has been registered alleging that {suspect_count} accused persons "
                    "jointly operated an ATM terminal in an unauthorised manner, one conducting the "
                    "transaction and the other ensuring no interference. {witness_count} witness(es) present."
                ),
                (
                    "CCTV review confirms that {suspect_count} suspects approached the ATM together. "
                    "Witness accounts describe clear role division between the accused. "
                    "{witness_count} independent observer(s) have provided corroborating statements."
                ),
                (
                    "Deponent reports being intimidated at the ATM by {suspect_count} accused who "
                    "worked in concert — one restraining the complainant while the other forced a "
                    "transaction. {witness_count} bystander(s) witnessed and reported the incident."
                ),
                (
                    "The investigating officer notes the presence of {suspect_count} accused at the "
                    "ATM location based on both CCTV evidence and eyewitness accounts. The coordinated "
                    "nature of the offence is confirmed by {witness_count} independent witness(es)."
                ),
            ],
        },
        time_window_range=(300, 900),
    ),

    "Office_Theft": DomainSpec(
        name="Office_Theft",
        crime_types=["office theft", "corporate document theft", "internal data breach"],
        locations=[
            "3rd Floor Office – Prestige Tech Park",
            "Admin Wing – Manyata Embassy Business Park",
            "Server Room – RMZ Ecospace",
            "Accounts Department – Brigade Gateway",
        ],
        template_descriptions={
            "Entry_HiddenAction_Exit": [
                (
                    "An employee reported missing items from the office premises after hours. "
                    "{suspect_count} individual(s) were identified on CCTV accessing the restricted "
                    "area without authorisation. {witness_count} colleague(s) provided supporting statements."
                ),
                (
                    "Security personnel flagged unauthorised entry into a restricted section of the office. "
                    "{suspect_count} unknown individual(s) were seen on premises outside working hours. "
                    "{witness_count} staff member(s) reported observing suspicious activity."
                ),
                (
                    "Physical documents and equipment reported missing from secure office area. "
                    "Access logs indicate entry by {suspect_count} unregistered individual(s) after hours. "
                    "{witness_count} witness(es) noted unusual presence on the floor."
                ),
                (
                    "The complainant reports that {suspect_count} unidentified individual(s) gained "
                    "unauthorised access to the office premises after closing hours and removed items. "
                    "{witness_count} staff member(s) corroborate observing the accused on the premises."
                ),
                (
                    "After-hours access logs flagged {suspect_count} entry event(s) by unregistered "
                    "personnel in a restricted section of the building. Items were found missing the "
                    "following morning. {witness_count} employee(s) have provided written statements."
                ),
                (
                    "Deponent states that {suspect_count} accused person(s) entered the office building "
                    "outside sanctioned working hours and removed confidential materials. "
                    "{witness_count} witness(es) observed the accused during the relevant time window."
                ),
                (
                    "Security footage confirms the presence of {suspect_count} unregistered individual(s) "
                    "in a restricted office zone after hours. The accused exited the premises with items "
                    "not logged through official channels. {witness_count} colleague(s) are on record."
                ),
                (
                    "A complaint has been registered regarding the theft of physical assets from the "
                    "office. CCTV evidence places {suspect_count} accused at the scene after hours. "
                    "{witness_count} staff member(s) have corroborated the timeline of events."
                ),
                (
                    "The investigating officer notes signs of forced or unauthorised access to "
                    "{suspect_count} restricted area(s) within the office complex. Missing items were "
                    "confirmed by inventory check. {witness_count} employee(s) provided eyewitness accounts."
                ),
                (
                    "Building access records confirm {suspect_count} badge swipe(s) into restricted "
                    "zones after official hours. Items were reported missing the following day. "
                    "{witness_count} office staff have corroborated the presence of the accused."
                ),
                (
                    "Complainant reports the disappearance of sensitive materials from a locked office "
                    "section. Surveillance confirms {suspect_count} individual(s) present at the "
                    "relevant time. {witness_count} witness(es) were in the vicinity and have filed statements."
                ),
                (
                    "The deponent alleges that {suspect_count} accused gained entry to the office "
                    "outside authorised hours using means yet to be established. Items of evidentiary "
                    "value are missing. {witness_count} staff witness(es) have confirmed the occurrence."
                ),
            ],
            "Entry_LegitAction_HiddenAction_Exit": [
                (
                    "An insider threat is suspected. A registered employee conducted normal work activities "
                    "during the day before accessing restricted files unusually late in their shift. "
                    "{witness_count} colleague(s) noted the individual staying back after others had left."
                ),
                (
                    "Confidential documents were found missing following an employee's shift. "
                    "Internal access logs show the registered user accessed restricted folders "
                    "beyond their clearance level. {witness_count} staff member(s) corroborate partial observations."
                ),
                (
                    "Data exfiltration suspected via external storage device. {suspect_count} employee(s) "
                    "were observed working late and departing with materials not checked out through "
                    "official channels. {witness_count} witness(es) have provided accounts."
                ),
                (
                    "Deponent reports that {suspect_count} employee(s) performed routine duties during "
                    "the day but subsequently accessed restricted systems well beyond their authorised "
                    "scope. {witness_count} colleague(s) noticed the unusual late-hour activity."
                ),
                (
                    "An internal audit flagged {suspect_count} registered user(s) who accessed "
                    "confidential files outside their designated clearance level. The employee had "
                    "been present for a legitimate shift earlier that day. {witness_count} colleague(s) are on record."
                ),
                (
                    "Complaint filed by the organisation regarding suspected data theft by "
                    "{suspect_count} employee(s) who exploited legitimate access credentials to "
                    "exfiltrate sensitive documents. {witness_count} co-worker(s) observed the accused "
                    "working unusually late."
                ),
                (
                    "The deponent alleges that {suspect_count} staff member(s) used their authorised "
                    "access to enter the premises under the guise of regular duties before transferring "
                    "restricted data to an external device. {witness_count} witness(es) corroborate the account."
                ),
                (
                    "System logs confirm that {suspect_count} registered employee(s) performed "
                    "legitimate clock-in activities before accessing files beyond their permission "
                    "level late in the shift. {witness_count} colleague(s) noticed and reported the conduct."
                ),
                (
                    "An IT security alert was triggered when {suspect_count} employee(s) attempted "
                    "to transfer large volumes of data to removable storage. The individual had "
                    "performed normal duties earlier in the day. {witness_count} staff member(s) are witnesses."
                ),
                (
                    "A registered staff member is suspected of using their routine office attendance "
                    "as cover for the theft of confidential materials. {witness_count} colleague(s) "
                    "report the accused remained on the premises well after closing time."
                ),
                (
                    "Deponent states that {suspect_count} insider(s) exploited legitimate system "
                    "access to extract proprietary data during a regular working shift. "
                    "{witness_count} co-worker(s) have confirmed the presence and conduct of the accused."
                ),
                (
                    "The investigating officer notes that access logs corroborate the deponent's "
                    "claim of insider data exfiltration by {suspect_count} registered user(s). "
                    "The accused had authorised entry earlier in the day. {witness_count} witness(es) on record."
                ),
            ],
        },
        time_window_range=(600, 86400),
    ),

    "Communication": DomainSpec(
        name="Communication",
        crime_types=[
            "criminal conspiracy via phone",
            "fraud coordination over email",
            "blackmail through digital channels",
        ],
        locations=[
            "Multiple Locations – Phone / Email",
            "Online – WhatsApp / Signal",
            "Corporate Email Server – Bengaluru",
        ],
        template_descriptions={
            "Communication_Action_Consequence": [
                (
                    "Intercepted communications suggest a single individual coordinated an illegal "
                    "activity via digital channels. A series of messages planning a transaction were "
                    "flagged by {witness_count} informant(s) who triggered the investigation."
                ),
                (
                    "A pattern of suspicious outbound communications was detected from {suspect_count} "
                    "individual(s). The messages contain coded references to planned illegal activity. "
                    "{witness_count} third party/parties reported receiving suspicious contact."
                ),
                (
                    "Digital forensics flagged {suspect_count} suspect(s) initiating and confirming "
                    "an illegal plan over encrypted channels. {witness_count} witness(es) inadvertently "
                    "intercepted or were copied on portions of the exchange."
                ),
                (
                    "The deponent reports receiving unsolicited communications from {suspect_count} "
                    "individual(s) that appear to relate to the planning of an unlawful transaction. "
                    "{witness_count} third party/parties independently flagged the same exchange."
                ),
                (
                    "Network monitoring systems flagged an unusual volume of encrypted messages "
                    "originating from {suspect_count} suspect(s). Content analysis suggests advance "
                    "planning of a criminal act. {witness_count} informant(s) corroborate the findings."
                ),
                (
                    "A complaint has been registered regarding digital communications of a criminal "
                    "nature attributed to {suspect_count} accused individual(s). "
                    "{witness_count} third party/parties who received or intercepted these messages "
                    "have provided written accounts."
                ),
                (
                    "The complainant alleges that {suspect_count} accused person(s) used digital "
                    "messaging platforms to solicit participation in an illegal scheme. "
                    "{witness_count} recipient(s) of the said communications have filed supporting statements."
                ),
                (
                    "Law enforcement received a tip-off from {witness_count} informant(s) regarding "
                    "communications attributed to {suspect_count} suspect(s) that indicate the "
                    "planning and confirmation of a criminal act via digital channels."
                ),
                (
                    "Call data records and message logs implicate {suspect_count} individual(s) "
                    "in the coordination of unlawful activity. {witness_count} person(s) who were "
                    "inadvertently included in the exchange have come forward as witnesses."
                ),
                (
                    "The investigating officer notes that digital evidence recovered from the "
                    "complainant's device shows communications from {suspect_count} accused "
                    "individual(s) arranging an illegal activity. {witness_count} corroborating witness(es) identified."
                ),
                (
                    "Deponent states that {suspect_count} accused person(s) made direct contact "
                    "via phone and messaging applications to coordinate what is alleged to be a "
                    "criminal conspiracy. {witness_count} independent party/parties have confirmed receipt of similar messages."
                ),
                (
                    "A forensic review of recovered devices reveals communications by {suspect_count} "
                    "suspect(s) consistent with the planning of an offence. "
                    "{witness_count} third party witness(es) have corroborated the existence and nature of the exchange."
                ),
            ],
            "MultiActor_Communication": [
                (
                    "Intercepted communications suggest coordinated activity among {suspect_count} "
                    "individual(s) operating across multiple channels. {witness_count} informant(s) "
                    "flagged the exchange pattern as consistent with pre-planned criminal coordination."
                ),
                (
                    "{suspect_count} suspects were found to be in active contact via encrypted messaging "
                    "platforms. The communication pattern — with multiple parties confirming roles and "
                    "timing — suggests organised criminal coordination. {witness_count} witness(es) alerted authorities."
                ),
                (
                    "A multi-party communication chain was identified involving {suspect_count} suspect(s) "
                    "exchanging operational details. {witness_count} third party/parties either intercepted "
                    "or were inadvertently included in the exchange, prompting the complaint."
                ),
                (
                    "The deponent alleges that {suspect_count} accused persons communicated repeatedly "
                    "via encrypted platforms to coordinate roles in a planned criminal act. "
                    "{witness_count} independent informant(s) flagged the exchange to investigating authorities."
                ),
                (
                    "Network analysis reveals a coordinated communication network involving "
                    "{suspect_count} suspect(s) exchanging messages across multiple digital channels. "
                    "{witness_count} person(s) intercepted portions of the exchange and reported it."
                ),
                (
                    "Law enforcement received intelligence indicating {suspect_count} accused individual(s) "
                    "were in active coordination over encrypted channels. {witness_count} informant(s) "
                    "provided corroborating evidence of the multi-party communication network."
                ),
                (
                    "A complaint was registered after {witness_count} informant(s) forwarded "
                    "communications to authorities implicating {suspect_count} individual(s) in "
                    "the organised planning of an illegal act over digital messaging platforms."
                ),
                (
                    "The investigating officer confirms that call detail records and messaging logs "
                    "establish active coordination between {suspect_count} accused persons. "
                    "{witness_count} third party witness(es) have independently corroborated the network of communications."
                ),
                (
                    "Deponent states that {suspect_count} accused individuals were identified as "
                    "participants in a digital communication chain that explicitly referenced the "
                    "planning of an offence. {witness_count} witness(es) have confirmed receipt of related messages."
                ),
                (
                    "A multi-accused communication conspiracy is alleged, with {suspect_count} "
                    "individual(s) using digital platforms to synchronise their roles. "
                    "{witness_count} independent observer(s) flagged the suspicious exchange pattern."
                ),
                (
                    "Forensic extraction of devices belonging to the accused confirms a sustained "
                    "communication network among {suspect_count} suspect(s). "
                    "{witness_count} informant(s) who received misdirected messages have provided statements."
                ),
                (
                    "The deponent alleges that {suspect_count} co-accused used a combination of "
                    "calls, messages, and encrypted channels to coordinate a criminal plan. "
                    "{witness_count} third party/parties intercepted or observed portions of these communications."
                ),
            ],
        },
        time_window_range=(3600, 86400),
    ),
}


def generate_fir(
    domain: str,
    template_name: str,
    suspect_count: int,
    witness_count: int,
    rng: random.Random,
) -> dict:
    """
    Generate a structured FIR dict for the given domain and template.

    The description is drawn from a template-specific pool so that the FIR
    reflects the actual nature of the crime (e.g. skimming vs. two-person
    robbery vs. insider theft) rather than being generic to the domain alone.

    Args:
        domain: One of the three supported domains.
        template_name: Name of the selected Template (used for description pool lookup).
        suspect_count: Number of suspects in the case.
        witness_count: Number of witnesses in the case.
        rng: Seeded random instance for reproducibility.

    Returns:
        FIR dict conforming to the fixed FIR schema.
    """
    spec = DOMAIN_SPECS[domain]
    crime_type = rng.choice(spec.crime_types)
    location = rng.choice(spec.locations)

    # Pick from template-specific descriptions; fall back to first available pool
    desc_pool = spec.template_descriptions.get(
        template_name,
        next(iter(spec.template_descriptions.values())),
    )
    description = rng.choice(desc_pool).format(
        suspect_count=suspect_count,
        witness_count=witness_count,
    )

    duration = rng.randint(*spec.time_window_range)

    return {
        "crime_type": crime_type,
        "location": location,
        "time_window": [0, duration],
        "description": description,
        "roles": {
            "suspect": suspect_count,
            "witness": witness_count,
        },
    }
