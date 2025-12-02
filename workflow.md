# AI Attendance System Workflow

This document describes the workflow for an AI-based attendance system that captures face scans, stores proof in the cloud, updates a database, and synchronizes attendance with ShikshaSetu ERP in real-time.

---

## Workflow Steps

1. **Face Scan & Photo Capture**
   - The system scans the user's face using a camera.
   - Upon successful recognition, a photo is captured instantly for proof.

2. **Cloud Storage Upload**
   - The captured photo is uploaded to secure cloud storage (e.g., AWS S3, Azure Blob, Google Cloud Storage).
   - The cloud returns a URL or reference to the stored image.

3. **Database Update**
   - Attendance data (user ID, timestamp, image URL) is saved to the database.
   - This ensures all attendance records are stored with proof.

4. **API Integration**
   - An API endpoint fetches the latest attendance data from the database.
   - The API prepares the data in the format required by ShikshaSetu ERP.

5. **Real-Time ERP Update**
   - The API sends the attendance data to ShikshaSetu ERP.
   - Attendance is updated in the ERP within seconds, ensuring instant record-keeping.

6. **Proof Availability**
   - The ERP or admin panel can access the cloud-stored photo as proof for each attendance entry.

---

## Workflow Diagram

```
[Face Scan & Photo Capture]
            |
            v
   [Upload Photo to Cloud]
            |
            v
   [Save Data to Database]
            |
            v
        [API Call]
            |
            v
[Update ShikshaSetu ERP (within seconds)]
            |
            v
   [Proof Accessible in ERP]
```

---

**Key Point:**
Attendance is updated in ShikshaSetu ERP within seconds, providing instant and reliable record-keeping with photographic proof.
