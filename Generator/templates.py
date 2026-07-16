"""
ForenSynth-X+ Templates
Event-sequence structure templates for each domain.
Each template defines an ordered list of abstract event slots that the
timeline generator will instantiate with concrete actors and actions.
"""

from dataclasses import dataclass
from typing import Literal

Domain = Literal["ATM_Robbery", "Office_Theft", "Communication"]


@dataclass
class EventSlot:
    """
    A single abstract slot in a template.

    Attributes:
        slot_id: Short identifier (e.g. "entry", "action_1").
        role: Which role class performs this action ("suspect" | "witness" | "system").
        role_index: 0-based index within the role class (None = any / first available).
        action: Canonical action verb used for ground-truth labelling.
        description_templates: List of surface-form strings for clean observation content.
        required: If False, slot may be skipped probabilistically.
        min_offset_delta: Minimum seconds after the previous event.
        max_offset_delta: Maximum seconds after the previous event.
    """
    slot_id: str
    role: str
    role_index: int | None
    action: str
    description_templates: list[str]
    required: bool = True
    min_offset_delta: int = 10
    max_offset_delta: int = 60


@dataclass
class Template:
    name: str
    domain: Domain
    slots: list[EventSlot]


# ---------------------------------------------------------------------------
# ATM_Robbery Templates
# ---------------------------------------------------------------------------

ATM_ENTRY_ACTION_EXIT = Template(
    name="Entry_Action_Exit",
    domain="ATM_Robbery",
    slots=[
        EventSlot(
            slot_id="approach",
            role="suspect", role_index=0,
            action="approach_atm",
            description_templates=[
                "Person approaches ATM from the street.",
                "Individual walks towards ATM booth.",
                "Suspect seen nearing the ATM kiosk.",
                "A figure is observed walking toward the ATM enclosure.",
                "Individual approaches the cash machine from the main road.",
                "Person seen heading in the direction of the ATM kiosk.",
                "Subject approaches the ATM location with apparent purpose.",
                "Individual seen moving towards the ATM from across the street.",
            ],
            min_offset_delta=5, max_offset_delta=30,
        ),
        EventSlot(
            slot_id="entry",
            role="suspect", role_index=0,
            action="enter_atm",
            description_templates=[
                "Person enters ATM booth.",
                "Individual pushes open ATM door and steps inside.",
                "Suspect seen entering the ATM enclosure.",
                "Subject steps into the ATM cabin and closes the door.",
                "Individual is observed entering the cash machine booth.",
                "Person pulls open the ATM door and proceeds inside.",
                "Suspect enters the ATM kiosk unaccompanied.",
                "Individual recorded crossing the threshold of the ATM enclosure.",
            ],
            min_offset_delta=10, max_offset_delta=40,
        ),
        EventSlot(
            slot_id="transaction",
            role="suspect", role_index=0,
            action="withdraw_cash",
            description_templates=[
                "Person interacts with ATM machine.",
                "Individual appears to operate the ATM terminal.",
                "Suspect conducting transaction at ATM.",
                "Subject is seen pressing keys on the ATM panel.",
                "Individual inserts card and operates the machine.",
                "Person engages with the ATM screen for an extended period.",
                "Suspect appears to complete a transaction at the ATM.",
                "Individual observed conducting activity at the cash machine.",
            ],
            min_offset_delta=20, max_offset_delta=120,
        ),
        EventSlot(
            slot_id="loiter_post",
            role="suspect", role_index=0,
            action="wait_outside",
            description_templates=[
                "Person lingers inside ATM booth after transaction.",
                "Individual seen standing near ATM after use.",
                "Subject remains in ATM enclosure after completing activity.",
                "Person does not immediately exit after using the machine.",
                "Individual observed loitering inside the booth post-transaction.",
                "Suspect delays exit from the ATM after conducting activity.",
                "Person stands near the ATM panel after the transaction concludes.",
                "Individual is seen waiting inside the ATM cabin without apparent reason.",
            ],
            min_offset_delta=10, max_offset_delta=40,
        ),
        EventSlot(
            slot_id="exit",
            role="suspect", role_index=0,
            action="exit_atm",
            description_templates=[
                "Person exits ATM booth.",
                "Individual leaves ATM enclosure.",
                "Suspect walks out of ATM.",
                "Subject is seen departing the ATM cabin.",
                "Individual exits through the ATM door and walks away.",
                "Person leaves the cash machine booth quickly.",
                "Suspect departs the ATM kiosk.",
                "Individual is recorded leaving the ATM enclosure.",
            ],
            min_offset_delta=10, max_offset_delta=50,
        ),
        EventSlot(
            slot_id="witness_observes",
            role="witness", role_index=0,
            action="observe_exit",
            description_templates=[
                "Bystander seen looking toward ATM and turning to observe departing individuals.",
                "Passerby stops walking and watches two people leave the ATM area.",
                "Person standing nearby turns their head as individuals exit the ATM.",
                "Bystander seen pointing toward the ATM area.",
                "Individual on pavement observed watching a person walk away from the ATM.",
                "Passerby stops and stares at individuals departing the ATM booth.",
                "Person standing at nearby bus stop watches individuals leave the ATM.",
                "Bystander seen looking in direction of ATM as subjects depart.",
            ],
            min_offset_delta=5, max_offset_delta=30,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Individual seen approaching a uniformed officer and gesturing toward the ATM.",
                "Person observed speaking on phone while pointing in direction of ATM.",
                "Bystander seen walking into a nearby bank branch.",
                "Individual seen typing on phone while standing near the ATM location.",
                "Person observed writing something on paper near the ATM area.",
                "Individual seen stopping a passing officer and pointing toward the ATM.",
                "Bystander observed speaking with a security guard outside the bank.",
                "Person seen dialling on mobile and speaking with visible urgency.",
            ],
            required=False,
            min_offset_delta=30, max_offset_delta=300,
        ),
    ],
)

ATM_MULTI_ACTOR_ENTRY_ACTION_EXIT = Template(
    name="MultiActor_Entry_Action_Exit",
    domain="ATM_Robbery",
    slots=[
        EventSlot(
            slot_id="suspect1_approach",
            role="suspect", role_index=0,
            action="approach_atm",
            description_templates=[
                "First individual approaches ATM.",
                "Person arrives near ATM booth.",
                "First suspect is seen nearing the ATM enclosure.",
                "Initial actor approaches the cash machine.",
                "One of the individuals is recorded walking towards the ATM.",
                "First person approaches ATM from the south entrance.",
                "Lead suspect moves toward the ATM kiosk.",
                "First actor arrives at the ATM location.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="suspect2_approach",
            role="suspect", role_index=1,
            action="approach_atm",
            description_templates=[
                "Second individual arrives near ATM.",
                "Another person approaches ATM from a different direction.",
                "A second figure approaches the ATM from the opposite side.",
                "Second actor arrives near the ATM booth shortly after the first.",
                "Another individual is seen walking towards the cash machine.",
                "Second suspect approaches the ATM from an adjacent street.",
                "A second person appears near the ATM enclosure.",
                "The second actor arrives at the ATM vicinity.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="suspect1_entry",
            role="suspect", role_index=0,
            action="enter_atm",
            description_templates=[
                "First suspect enters ATM booth.",
                "Individual steps inside ATM enclosure.",
                "First actor enters the ATM cabin.",
                "Lead suspect pushes into the ATM enclosure.",
                "First individual steps through the ATM booth door.",
                "Primary suspect enters the cash machine booth.",
                "First person proceeds into the ATM kiosk.",
                "Lead actor enters the ATM enclosure and closes the door.",
            ],
            min_offset_delta=10, max_offset_delta=30,
        ),
        EventSlot(
            slot_id="suspect2_wait",
            role="suspect", role_index=1,
            action="wait_outside",
            description_templates=[
                "Second person loiters near ATM entrance.",
                "Individual stands guard at ATM door.",
                "Second suspect positions themselves at the ATM entrance.",
                "Second actor remains outside the booth, watching the area.",
                "The second individual stands near the ATM door.",
                "Second suspect acts as lookout near the ATM entrance.",
                "Second person waits at the threshold of the ATM cabin.",
                "Second actor stands near the ATM entrance, facing outward.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="transaction",
            role="suspect", role_index=0,
            action="withdraw_cash",
            description_templates=[
                "Person inside ATM conducts transaction.",
                "ATM machine is operated by individual inside booth.",
                "Individual inside the ATM booth interacts with the terminal.",
                "First suspect conducts activity at the ATM panel.",
                "Person inside the cabin is seen pressing ATM keys.",
                "Lead suspect operates the ATM machine inside the booth.",
                "Individual inside the kiosk appears to complete a transaction.",
                "First actor interfaces with the ATM terminal inside the booth.",
            ],
            min_offset_delta=30, max_offset_delta=120,
        ),
        EventSlot(
            slot_id="suspect1_exit",
            role="suspect", role_index=0,
            action="exit_atm",
            description_templates=[
                "First suspect exits ATM booth.",
                "Individual leaves ATM enclosure.",
                "Lead suspect departs the ATM cabin.",
                "First actor exits through the ATM door.",
                "Primary suspect walks out of the ATM enclosure.",
                "First individual leaves the cash machine booth.",
                "Lead actor exits the ATM kiosk and rejoins the second person.",
                "First suspect is seen leaving the ATM enclosure.",
            ],
            min_offset_delta=10, max_offset_delta=30,
        ),
        EventSlot(
            slot_id="both_flee",
            role="suspect", role_index=1,
            action="flee_scene",
            description_templates=[
                "Both individuals leave the ATM area together.",
                "Two persons seen walking away rapidly.",
                "Both suspects depart the ATM vicinity together.",
                "Two individuals are recorded leaving the scene side by side.",
                "Both actors walk away from the ATM area quickly.",
                "The two suspects are seen fleeing together.",
                "Both persons leave the ATM location at pace.",
                "Two figures are seen departing the ATM vicinity in unison.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="witness_observes",
            role="witness", role_index=0,
            action="observe_exit",
            description_templates=[
                "Bystander on pavement seen watching two individuals leave the ATM.",
                "Person nearby stops and turns to observe two people departing the ATM.",
                "Bystander seen staring at two people walking away from the ATM.",
                "Passerby pauses and watches two individuals move away from the ATM area.",
                "Individual near ATM seen craning neck to watch two people leave.",
                "Person at bus stop observed watching two individuals walk away quickly.",
                "Bystander seen standing still and tracking two departing individuals.",
                "Person near shop entrance observed staring at two people leaving the ATM.",
            ],
            required=False,
            min_offset_delta=5, max_offset_delta=25,
        ),
        EventSlot(
            slot_id="witness2_reports",
            role="witness", role_index=1,
            action="report_incident",
            description_templates=[
                "Second individual seen approaching a police officer near the ATM.",
                "Another person observed typing urgently on a phone near the ATM.",
                "Second bystander seen walking toward a bank security guard.",
                "Another person seen stopping a uniformed officer and pointing.",
                "Second individual observed entering a nearby bank branch.",
                "Another bystander seen speaking with a security guard outside.",
                "Second person observed dialling on phone near the ATM area.",
                "Another individual seen gesturing toward the ATM while on a call.",
            ],
            required=False,
            min_offset_delta=30, max_offset_delta=300,
        ),
    ],
)

ATM_ENTRY_SUSPICIOUS_EXIT = Template(
    name="Entry_Suspicious_Exit",
    domain="ATM_Robbery",
    slots=[
        EventSlot(
            slot_id="approach",
            role="suspect", role_index=0,
            action="approach_atm",
            description_templates=[
                "Person approaches ATM from the street.",
                "Individual seen walking towards ATM kiosk.",
                "Subject is seen nearing the ATM from the main road.",
                "Individual approaches the cash machine with apparent intent.",
                "Person observed walking in the direction of the ATM enclosure.",
                "Suspect seen approaching ATM from across the street.",
                "A figure approaches the ATM kiosk.",
                "Individual moves toward the ATM booth from a nearby location.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="loiter",
            role="suspect", role_index=0,
            action="loiter_near_atm",
            description_templates=[
                "Person seen pacing near ATM for extended period.",
                "Individual observed loitering outside ATM booth.",
                "Subject lingers near the ATM kiosk for an unusually long time.",
                "A person is seen standing near the ATM without transacting.",
                "Individual hovers near the cash machine entrance for several minutes.",
                "Suspect is seen walking back and forth near the ATM repeatedly.",
                "Person loiters in the vicinity of the ATM without apparent purpose.",
                "Subject observed near ATM for extended duration before entry.",
            ],
            min_offset_delta=30, max_offset_delta=180,
        ),
        EventSlot(
            slot_id="entry",
            role="suspect", role_index=0,
            action="enter_atm",
            description_templates=[
                "Suspect enters ATM booth.",
                "Person pushes into ATM enclosure.",
                "Individual steps into the ATM cabin after loitering.",
                "Suspect enters the cash machine booth.",
                "Person proceeds into the ATM kiosk.",
                "Subject enters the ATM enclosure after surveying the area.",
                "Individual pushes open the ATM door and goes inside.",
                "Suspect enters the ATM booth unaccompanied.",
            ],
            min_offset_delta=10, max_offset_delta=30,
        ),
        EventSlot(
            slot_id="suspicious_action",
            role="suspect", role_index=0,
            action="tamper_atm",
            description_templates=[
                "Person appears to tamper with ATM card slot.",
                "Individual seen attaching device to ATM machine.",
                "Suspect spends unusual amount of time at ATM panel.",
                "Subject is observed manipulating the card reader mechanism.",
                "Individual seen fiddling with the ATM card insertion area.",
                "Suspect appears to affix a foreign device to the ATM slot.",
                "Person makes prolonged contact with the ATM card reader.",
                "Individual observed handling the ATM panel in a suspicious manner.",
            ],
            min_offset_delta=20, max_offset_delta=90,
        ),
        EventSlot(
            slot_id="exit",
            role="suspect", role_index=0,
            action="exit_atm",
            description_templates=[
                "Person exits ATM quickly.",
                "Individual leaves ATM booth.",
                "Suspect departs the ATM enclosure hastily.",
                "Person is seen leaving the ATM cabin in a hurry.",
                "Individual exits the ATM booth and walks off.",
                "Suspect leaves the cash machine booth abruptly.",
                "Person departs from the ATM kiosk at pace.",
                "Subject exits the ATM enclosure without completing any transaction.",
            ],
            min_offset_delta=5, max_offset_delta=20,
        ),
        EventSlot(
            slot_id="witness_observes",
            role="witness", role_index=0,
            action="observe_exit",
            description_templates=[
                "Bystander notices person leaving ATM area hastily.",
                "Witness sees individual exit ATM and walk away.",
                "Passerby observes suspect departing ATM quickly.",
                "Eyewitness notices a person leaving the ATM enclosure in a rush.",
                "A bystander sees someone exit the ATM and walk away briskly.",
                "Witness spots individual departing the cash machine area suddenly.",
                "Onlooker observes a person leaving the ATM kiosk abruptly.",
                "Bystander seen turning to track individual walking away from ATM.",
            ],
            min_offset_delta=5, max_offset_delta=25,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Individual seen approaching a uniformed officer near the ATM area.",
                "Person observed typing on phone with visible urgency near the ATM.",
                "Bystander seen walking into a nearby establishment while looking back at ATM.",
                "Individual observed speaking to another person and gesturing toward ATM.",
                "Person observed walking to a bank security guard near the entrance.",
                "Bystander seen approaching a uniformed officer near the ATM.",
                "Bystander seen speaking with bank security personnel outside.",
                "Person observed writing in a notebook while looking toward the ATM.",
            ],
            required=False,
            min_offset_delta=60, max_offset_delta=600,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Office_Theft Templates
# ---------------------------------------------------------------------------

OFFICE_ENTRY_HIDDEN_ACTION_EXIT = Template(
    name="Entry_HiddenAction_Exit",
    domain="Office_Theft",
    slots=[
        EventSlot(
            slot_id="approach_building",
            role="suspect", role_index=0,
            action="approach_office",
            description_templates=[
                "Individual seen approaching office building.",
                "Person arrives at building entrance.",
                "Subject is observed arriving at the office premises.",
                "Individual approaches the office building from the parking area.",
                "Person is seen nearing the office entrance.",
                "Suspect is observed walking toward the building lobby.",
                "Individual approaches the office block.",
                "Person seen heading toward the office building entrance.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="entry",
            role="suspect", role_index=0,
            action="enter_office",
            description_templates=[
                "Person enters office premises after hours.",
                "Individual seen accessing office building.",
                "Suspect uses access card to enter restricted area.",
                "Subject is recorded entering the office outside working hours.",
                "Individual gains entry to the building using a badge.",
                "Person enters the office premises without authorisation.",
                "Suspect is seen swiping an access card and entering the building.",
                "Individual enters the office block after closing time.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="navigate",
            role="suspect", role_index=0,
            action="navigate_to_target",
            description_templates=[
                "Individual moves towards the accounts section.",
                "Person walks through corridor towards server room.",
                "Suspect seen navigating office floor.",
                "Subject is seen walking through the office corridor.",
                "Individual makes their way to the restricted floor.",
                "Person is recorded moving toward the accounts department.",
                "Suspect walks through the office hallway toward the target area.",
                "Individual is seen navigating toward the server room.",
            ],
            min_offset_delta=20, max_offset_delta=120,
        ),
        EventSlot(
            slot_id="theft",
            role="suspect", role_index=0,
            action="steal_items",
            description_templates=[
                "Person seen handling files/documents.",
                "Individual accesses drawer and removes contents.",
                "Suspect observed interacting with computer system.",
                "Subject is seen removing physical files from a cabinet.",
                "Individual opens a restricted drawer and takes documents.",
                "Person is recorded interacting with a workstation in a restricted zone.",
                "Suspect handles folders and places them in a bag.",
                "Individual removes items from the office and conceals them.",
            ],
            min_offset_delta=30, max_offset_delta=300,
        ),
        EventSlot(
            slot_id="exit",
            role="suspect", role_index=0,
            action="exit_office",
            description_templates=[
                "Person exits office building.",
                "Individual leaves through emergency exit.",
                "Suspect seen departing office premises.",
                "Subject exits the building through a side door.",
                "Individual leaves the office floor and heads for the exit.",
                "Person departs the office building carrying a bag.",
                "Suspect exits the premises via the stairwell.",
                "Individual is seen leaving the building after hours.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="witness_observes",
            role="witness", role_index=0,
            action="observe_exit",
            description_templates=[
                "Colleague seen pausing in corridor to look at an unknown person.",
                "Staff member seen stopping and watching unfamiliar individual on restricted floor.",
                "Co-worker seen turning to observe a person leaving the server room area.",
                "Colleague observed holding a door open and looking down the corridor.",
                "Staff member seen standing still and watching an unknown person walk past.",
                "Co-worker seen looking up from workstation as unknown individual passes.",
                "Colleague observed watching a person enter an area marked restricted.",
                "Staff member seen doing a double-take as unfamiliar person exits a secure area.",
            ],
            min_offset_delta=20, max_offset_delta=180,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Colleague seen walking to the security desk and speaking to a guard.",
                "Staff member observed approaching the reception desk.",
                "Co-worker seen stopping at the security station and gesturing.",
                "Colleague observed speaking to another staff member near the lift.",
                "Employee seen approaching a manager's office and knocking.",
                "Staff member observed typing at a terminal with visible concern.",
                "Co-worker seen walking quickly to the IT help desk.",
                "Colleague observed making a call while looking toward the restricted area.",
            ],
            required=False,
            min_offset_delta=300, max_offset_delta=3600,
        ),
    ],
)

OFFICE_LEGIT_THEN_HIDDEN = Template(
    name="Entry_LegitAction_HiddenAction_Exit",
    domain="Office_Theft",
    slots=[
        EventSlot(
            slot_id="approach_building",
            role="suspect", role_index=0,
            action="approach_office",
            description_templates=[
                "Employee arrives at office building entrance.",
                "Individual seen approaching office premises.",
                "Registered staff member arrives at the building as normal.",
                "Employee approaches office entrance at the start of shift.",
                "Individual seen arriving at the office during business hours.",
                "Staff member is recorded arriving at the office premises.",
                "Employee enters the building at their normal start time.",
                "Individual arrives at the office block for their shift.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="entry",
            role="suspect", role_index=0,
            action="enter_office",
            description_templates=[
                "Employee enters office building during working hours.",
                "Individual arrives at office and badges in.",
                "Staff member swipes their access card and enters the building.",
                "Employee enters the office premises during their authorised shift.",
                "Individual badges into the building as part of normal routine.",
                "Registered employee enters the office block during working hours.",
                "Staff member enters the building using their valid access credentials.",
                "Employee is seen entering the office floor during their shift.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="legit_work",
            role="suspect", role_index=0,
            action="perform_legit_work",
            description_templates=[
                "Employee works at desk, conducting normal activities.",
                "Individual participates in team meeting.",
                "Person seen working at computer terminal normally.",
                "Staff member is observed carrying out regular work tasks.",
                "Individual is seen at their workstation performing normal duties.",
                "Employee participates in scheduled meetings during the day.",
                "Person is recorded performing standard job functions at their desk.",
                "Staff member works at their computer without apparent irregularity.",
            ],
            min_offset_delta=1800, max_offset_delta=7200,
        ),
        EventSlot(
            slot_id="hidden_action",
            role="suspect", role_index=0,
            action="steal_data",
            description_templates=[
                "Person transfers files to external storage device.",
                "Individual accesses restricted folder late in day.",
                "Suspect seen copying documents to USB drive.",
                "Subject is recorded connecting a USB device and initiating a file transfer.",
                "Individual accesses a folder outside their authorised permissions.",
                "Employee is seen copying large volumes of data late in their shift.",
                "Person exports restricted files to an external storage medium.",
                "Suspect is observed transferring data to a removable device.",
            ],
            min_offset_delta=60, max_offset_delta=600,
        ),
        EventSlot(
            slot_id="exit",
            role="suspect", role_index=0,
            action="exit_office",
            description_templates=[
                "Employee leaves office at end of day.",
                "Individual exits building with bag.",
                "Staff member departs the office at the end of their shift.",
                "Individual is seen leaving the building with a bag.",
                "Employee exits the office premises with personal items.",
                "Person leaves the office floor and heads to the exit.",
                "Staff member departs the building later than their usual time.",
                "Individual is recorded leaving the office with a bag not present on arrival.",
            ],
            min_offset_delta=10, max_offset_delta=60,
        ),
        EventSlot(
            slot_id="witness_observes",
            role="witness", role_index=0,
            action="observe_exit",
            description_templates=[
                "Colleague notices individual leaving with unusual amount of items.",
                "Witness spots employee departing at unusual time.",
                "Co-worker observes the individual carrying more than usual on departure.",
                "Colleague notices the employee leaving much later than normal.",
                "Staff member observes the suspect carrying an oversized bag on exit.",
                "Witness sees the employee leaving the floor at an unusual hour.",
                "Co-worker notes the individual departing in an uncharacteristic hurry.",
                "Colleague observes the employee exiting via an infrequently used route.",
            ],
            min_offset_delta=5, max_offset_delta=120,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Colleague seen speaking to another staff member in the corridor.",
                "Employee observed approaching the security desk near the exit.",
                "Co-worker seen walking to the IT help desk with a folder.",
                "Staff member observed typing on a terminal with visible urgency.",
                "Colleague seen entering a manager's office and closing the door.",
                "Employee observed speaking to a guard near the building exit.",
                "Co-worker seen approaching reception with a printed document.",
                "Staff member observed in animated conversation near the HR office.",
            ],
            required=False,
            min_offset_delta=300, max_offset_delta=7200,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Communication Templates
# ---------------------------------------------------------------------------

COMM_ACTION_CONSEQUENCE = Template(
    name="Communication_Action_Consequence",
    domain="Communication",
    slots=[
        EventSlot(
            slot_id="initiate_comm",
            role="suspect", role_index=0,
            action="initiate_communication",
            description_templates=[
                "Individual seen holding a phone to their ear.",
                "Person observed using a mobile device.",
                "Subject seen looking at a phone screen and typing.",
                "Individual recorded placing a call while seated.",
                "Person is seen with a phone raised to their ear.",
                "Subject observed holding a handset and speaking.",
                "Individual sits and looks at a mobile device for an extended period.",
                "Person observed using a phone in a public area.",
            ],
            min_offset_delta=60, max_offset_delta=3600,
        ),
        EventSlot(
            slot_id="exchange",
            role="suspect", role_index=0,
            action="exchange_information",
            description_templates=[
                "Individual observed using a mobile device for an extended period.",
                "Person seen repeatedly looking at a phone screen.",
                "Subject recorded typing on a mobile device.",
                "Individual seen with phone in hand, alternating between screen and surroundings.",
                "Person sits with a mobile device, appearing to read and respond.",
                "Subject observed typing on a handheld device at a table.",
                "Individual seen with phone face-down after using it.",
                "Person observed using a mobile device in a parked vehicle.",
            ],
            min_offset_delta=120, max_offset_delta=1800,
        ),
        EventSlot(
            slot_id="plan_confirmed",
            role="suspect", role_index=0,
            action="confirm_plan",
            description_templates=[
                "Individual puts phone away after a brief exchange.",
                "Person seen nodding while on a call, then ending it.",
                "Subject is observed sending a message and pocketing the device.",
                "Individual concludes a phone call and stands up.",
                "Person sends a message, then closes the phone screen.",
                "Subject places phone on the table face-down after a brief call.",
                "Individual ends a call and immediately leaves the area.",
                "Person seen typing a short message then setting the device down.",
            ],
            min_offset_delta=300, max_offset_delta=3600,
        ),
        EventSlot(
            slot_id="suspect_follow_up",
            role="suspect", role_index=0,
            action="coordinate_activity",
            description_templates=[
                "Individual picks up phone again and places a second call.",
                "Person seen sending additional messages on a mobile device.",
                "Subject observed making another call shortly after the first.",
                "Individual seen looking at phone and typing repeatedly.",
                "Person uses phone a second time within a short interval.",
                "Subject makes a follow-up call and hangs up quickly.",
                "Individual seen typing on mobile device for a second time.",
                "Person checks phone and then places a brief call.",
            ],
            min_offset_delta=300, max_offset_delta=7200,
        ),
        EventSlot(
            slot_id="witness_flags",
            role="witness", role_index=0,
            action="flag_suspicious_comm",
            description_templates=[
                "Individual seen looking at phone screen with visible surprise.",
                "Person observed staring at a received message for an extended period.",
                "Individual seen showing phone screen to another person nearby.",
                "Person observed typing rapidly after reading a received message.",
                "Individual seen taking a screenshot on a mobile device.",
                "Person observed putting phone down after reading a message and looking unsettled.",
                "Individual seen typing and then looking around with visible concern.",
                "Person observed forwarding a message and then making a phone call.",
            ],
            min_offset_delta=1800, max_offset_delta=86400,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Individual seen approaching a uniformed officer with phone in hand.",
                "Person observed speaking to someone in a formal setting while holding a phone.",
                "Individual seen placing phone on a table and sliding it toward another person.",
                "Person observed speaking and gesturing with phone visible.",
                "Individual seen typing on phone before walking toward a government office.",
                "Person observed handing a device to an official across a desk.",
                "Individual seen in conversation with an officer while pointing at phone screen.",
                "Person observed entering a police station with a document in hand.",
            ],
            min_offset_delta=600, max_offset_delta=7200,
        ),
    ],
)

COMM_MULTI_ACTOR = Template(
    name="MultiActor_Communication",
    domain="Communication",
    slots=[
        EventSlot(
            slot_id="suspect1_initiate",
            role="suspect", role_index=0,
            action="initiate_communication",
            description_templates=[
                "First individual seen holding a phone and speaking.",
                "Primary person observed using a mobile device.",
                "First individual recorded placing a call while seated.",
                "Lead person seen with phone raised, speaking into handset.",
                "First individual observed typing on a mobile device.",
                "Primary actor seen with phone in hand at a table.",
                "First person observed dialling on a mobile phone.",
                "Lead individual uses a phone and appears to be in conversation.",
            ],
            min_offset_delta=60, max_offset_delta=1800,
        ),
        EventSlot(
            slot_id="suspect2_respond",
            role="suspect", role_index=1,
            action="respond_to_communication",
            description_templates=[
                "Second individual seen looking at a mobile phone screen.",
                "Second person observed typing a reply on a mobile device.",
                "Second individual seen answering a call.",
                "Second actor observed holding phone to ear briefly.",
                "Second person seen reading a message and then typing.",
                "Second individual picks up phone and appears to respond.",
                "Second actor seen with phone in hand, scrolling the screen.",
                "Second person observed sending a short message and pocketing phone.",
            ],
            min_offset_delta=300, max_offset_delta=3600,
        ),
        EventSlot(
            slot_id="coordinate",
            role="suspect", role_index=0,
            action="coordinate_activity",
            description_templates=[
                "Individual seen using phone again shortly after a previous call.",
                "Person observed making a second call within a brief interval.",
                "Subject seen typing repeatedly on a mobile device.",
                "Individual picks up phone and makes another brief call.",
                "Person observed alternating between screen and surroundings repeatedly.",
                "Individual observed with phone, appears to check and respond.",
                "Person makes a further call and gestures while speaking.",
                "Individual uses phone a second time within the same location.",
            ],
            min_offset_delta=300, max_offset_delta=7200,
        ),
        EventSlot(
            slot_id="suspect2_confirm",
            role="suspect", role_index=1,
            action="confirm_plan",
            description_templates=[
                "Second individual seen sending a brief message on a mobile device.",
                "Second person observed typing on a phone and then pocketing it.",
                "Second individual seen looking at phone and nodding before putting it away.",
                "Second person observed ending a call and setting the device down.",
                "Second individual seen typing a short message and then standing up.",
                "Second actor observed with phone in hand, briefly typing.",
                "Second person seen checking a message and responding with a short text.",
                "Second individual observed placing a call and speaking briefly.",
            ],
            min_offset_delta=300, max_offset_delta=3600,
        ),
        EventSlot(
            slot_id="witness_intercept",
            role="witness", role_index=0,
            action="intercept_communication",
            description_templates=[
                "Individual receives an unexpected message on their device.",
                "Person seen looking at a received message with visible confusion.",
                "Third party seen showing a phone screen to another person.",
                "Individual observed with phone, reviewing a message.",
                "Person seen taking a screenshot and forwarding a message.",
                "Individual observed setting phone down after reading a message.",
                "Person seen typing a forwarded message to another contact.",
                "Individual observed staring at a received message for an extended period.",
            ],
            min_offset_delta=600, max_offset_delta=14400,
        ),
        EventSlot(
            slot_id="witness_reports",
            role="witness", role_index=0,
            action="report_incident",
            description_templates=[
                "Individual seen approaching a police officer and handing over a phone.",
                "Person observed seated across from an official, sliding a device over.",
                "Individual seen entering a government building with a document.",
                "Person observed in conversation with a uniformed officer, showing phone screen.",
                "Individual seen typing on phone before entering a police station.",
                "Person observed pointing at a phone screen while speaking to a security officer.",
                "Individual seen handing a printed document to a person in uniform.",
                "Person observed speaking urgently to an officer near a security desk.",
            ],
            min_offset_delta=600, max_offset_delta=7200,
        ),
    ],
)


# Registry: domain → list of templates
TEMPLATE_REGISTRY: dict[str, list[Template]] = {
    "ATM_Robbery": [
        ATM_ENTRY_ACTION_EXIT,
        ATM_MULTI_ACTOR_ENTRY_ACTION_EXIT,
        ATM_ENTRY_SUSPICIOUS_EXIT,
    ],
    "Office_Theft": [
        OFFICE_ENTRY_HIDDEN_ACTION_EXIT,
        OFFICE_LEGIT_THEN_HIDDEN,
    ],
    "Communication": [
        COMM_ACTION_CONSEQUENCE,
        COMM_MULTI_ACTOR,
    ],
}


def select_template(domain: str, rng: "random.Random") -> Template:
    """Randomly select a template for the given domain."""
    import random
    options = TEMPLATE_REGISTRY[domain]
    return rng.choice(options)


def get_template_role_requirements(template: Template) -> dict[str, int]:
    """
    Extract minimum role counts required by the template.
    
    Scans all slots and finds the maximum role_index for each role.
    The minimum count needed is max_index + 1 (since indices are 0-based).
    
    Args:
        template: The Template object to analyze.
        
    Returns:
        Dict mapping role name → minimum count required.
        E.g., {"suspect": 3, "witness": 1, "system": 0}
        
    Example:
        If template has slots with:
          - role="suspect", role_index=0
          - role="suspect", role_index=1
          - role="witness", role_index=0
        Then returns {"suspect": 2, "witness": 1}
    """
    role_max_index: dict[str, int] = {}
    
    for slot in template.slots:
        if slot.role_index is not None:
            current_max = role_max_index.get(slot.role, -1)
            role_max_index[slot.role] = max(current_max, slot.role_index)
    
    # Convert from max_index to required count (count = max_index + 1)
    return {role: (max_idx + 1) for role, max_idx in role_max_index.items()}
