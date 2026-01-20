# DRUGS.COM HB BID REQUEST

## Complete Bid Request JSON

```json
[
  {
    "id": "c00d356b-fbc1-4326-9fa2-d60d137a041f",
    "imp": [
      {
        "id": "62f396e130cbbed",
        "bidfloor": 0,
        "banner": {
          "h": 250,
          "w": 300
        }
      }
    ],
    "placementId": 111563,
    "site": {
      "page": "https://www.drugs.com/alpha/d.html",
      "name": "www.drugs.com"
    },
    "user": {
      "id": "",
      "buyeruid": "",
      "keywords": "",
      "customdata": "",
      "ext": {
        "eids": [
          {
            "source": "liveramp.com",
            "uids": [
              {
                "id": "AkZXUsXUHcpx5417c7yfT5j7ExZ5-E2pIvGfKchqGPQRQsHu6UNB-NT8N0S3Jn-qWSqWmfE-MyP501LAT2Afri-LVEp2LvLuNbPRI82EZGMzpP-h",
                "atype": 3
              }
            ]
          },
          {
            "source": "drugs.com",
            "uids": [
              {
                "id": "afe20b8fe2d1174c77a4658c5cbf3ef4c7a9029722d11875930555f6b19c4e98",
                "atype": 3,
                "ext": {
                  "stype": "ppuid"
                }
              }
            ]
          },
          {
            "source": "pubcid.org",
            "uids": [
              {
                "id": "ae18ec4e-2610-48ff-9f65-573b3c13e63d",
                "atype": 1
              }
            ]
          }
        ]
      }
    },
    "MediaType": "banner"
  }
]
```

---

## Field Descriptions

### Top-Level

- **`id`**: `"c00d356b-fbc1-4326-9fa2-d60d137a041f"`
  → Unique request ID for this auction call.

- **`placementId`**: `111563`
  → Internal placement/slot ID configured in the header bidder for this ad slot.

- **`MediaType`**: `"banner"`
  → Specifies this is a banner display ad request (not native/video).

---

### imp (Impression Object)

Defines the ad opportunity:

- **`id`**: `"62f396e130cbbed"`
  → Impression ID, unique for this slot.

- **`bidfloor`**: `0`
  → No minimum floor price (open to any bids).

- **`banner`**:
  - **`h`**: `250`
  - **`w`**: `300`
  → Banner ad size = 300×250 px (Medium Rectangle / MPU).

---

### site

Info about the publisher page:

- **`page`**: `"https://www.drugs.com/alpha/d.html"`
  → The exact page where the ad slot is located (Alphabetical index "D").

- **`name`**: `"www.drugs.com"`
  → Publisher site name.

---

### user

User-related identifiers and IDs passed for targeting.

- **`id`**: `""` (empty)
  → No exchange-level user ID provided.

- **`buyeruid`**: `""` (empty)
  → No DSP-mapped user ID.

- **`keywords`** / **`customdata`**: `""`
  → Empty, nothing passed here.

- **`ext.eids`**: Enriched identifiers, very important for ID-based targeting:
  - **source** = `liveramp.com`
    - **uid** = long encrypted string
    - **atype** = `3` → Cross-device / deterministic ID.
  - **source** = `drugs.com`
    - **uid** = hashed ID from publisher's own system
    - **stype** = `ppuid` → Publisher Provided User ID.
  - **source** = `pubcid.org`
    - **uid** = `ae18ec4e-2610-48ff-9f65-573b3c13e63d`
    - **atype** = `1` → Cookie-based ID.
