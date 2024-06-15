# Anterior Assessment

Welcome to the AI technical assessment for the position of AI Engineer at Anterior.

We quantify the value of our products by the value they provide to our end users, being a customer-driven company.


# Background
Included in this task is a redacted medical record for a patient staying at a hospital. While in the hospital, 
the clinicians will review the patient's case and determine whether action is required e.g. move to intensive care 
or subscribe to a new medication. The cadence of which the patient is reviewed is proportional to the severity of the 
condition and the current cost of care i.e. patients in intensive care are reviewed more frequently than patients on 
a lower cost / more relaxed ward.

Each time action is undertaken, this record is updated. This newly updated record is then compared against a set of 
guidelines that outline the valid insurance claims, given the patient's updated condition.

# Task
We require a temporally aware representation of any useful information within this record. Be aware that
1. duplicates may occur and the same information might appear on multiple pages. 
2. pages are not necessarily ordered chronologically

### Acceptance Criteria
1. Complete the TODO in `submission.py` so we can run your pipeline on our held-out test set

### Additional Criteria
1. We are seeking candidates who are deeply aware of the current LLM ecosystems.
2. If an opportunity presents itself, we would like you to teach us something new.

### Useful Resources
1. We have included a `src` package containing some helper code for your convenience.
2. Watch [this Loom](https://www.loom.com/share/4d4ee611b4504cb7977cb47a9fc0058c?sid=db01e5aa-e057-4fa1-af85-94090c7f0c9d) for a 5 minute GCP setup tutorial. 


# Submission
1. Fork this repository
2. Add a SUBMISSION.md which outlines your approach and motivation. We value communication skills.
3. Attach a link to a screen recording of you demoing your submission. 
   1. Hard limit of 8 minutes. 
   2. Ensure that the sharing link has general viewer permissions.
4. Upload your submission to Github and share the repo link with your Anterior point of contact. 
