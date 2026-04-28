# Comprehensive OWASP Top 10 CWE Mappings (2025, 2021, 2017)

This document contains the comprehensive list of Common Weakness Enumerations (CWEs) mapped to each category for the 2025, 2021, and 2017 editions of the OWASP Top 10.

---

## 1. OWASP Top 10:2025 (Current Edition)

### A01:2025 – Broken Access Control
*Consolidated to include SSRF.*
* CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
* CWE-23: Relative Path Traversal
* CWE-24: Path Traversal: '../filedir'
* CWE-25: Path Traversal: '/../filedir'
* CWE-26: Path Traversal: '/dir/../filename'
* CWE-27: Path Traversal: 'dir/../../filename'
* CWE-28: Path Traversal: '..iledir'
* CWE-29: Path Traversal: '\..ilename'
* CWE-30: Path Traversal: '\..ilename'
* CWE-31: Path Traversal: 'dir\..\..ilename'
* CWE-32: Path Traversal: '...' (Triple Dot)
* CWE-33: Path Traversal: '....' (Multiple Dot)
* CWE-34: Path Traversal: '....//'
* CWE-35: Path Traversal: '.../...//'
* CWE-55: Path Equivalence: '/./' (Single Dot Directory)
* CWE-56: Path Equivalence: 'filedir*' (Wildcard)
* CWE-57: Path Equivalence: 'fakedir/../realdir/filename'
* CWE-58: Path Equivalence: Windows 8.3 Filename
* CWE-59: Improper Link Resolution Before File Access ('Link Following')
* CWE-60: UNIX Path Equivalence: Single Quote / Double Quote
* CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
* CWE-219: Sensitive Data Under Web Root
* CWE-264: Permissions, Privileges, and Access Controls
* CWE-275: Permission Issues
* CWE-276: Incorrect Default Permissions
* CWE-284: Improper Access Control
* CWE-285: Improper Authorization
* CWE-352: Cross-Site Request Forgery (CSRF)
* CWE-359: Exposure of Private Personal Information to an Unauthorized Actor
* CWE-434: Unrestricted Upload of File with Dangerous Type
* CWE-552: Files or Directories Accessible to External Parties
* CWE-639: Authorization Bypass Through User-Controlled Key
* CWE-732: Incorrect Permission Assignment for Critical Resource
* CWE-862: Missing Authorization
* CWE-863: Incorrect Authorization
* CWE-918: Server-Side Request Forgery (SSRF)

### A02:2025 – Security Misconfiguration
* CWE-16: Configuration
* CWE-209: Generation of Error Messages Containing Sensitive Information
* CWE-215: Information Exposure Through Debug Information
* CWE-237: Improper Handling of Structural Elements
* CWE-383: J2EE Bad Practices: Incomplete custom error handling
* CWE-536: Servlet Runtime Error Message Containing Sensitive Information
* CWE-548: Exposure of Information Through Directory Listing
* CWE-611: Improper Restriction of XML External Entity Reference
* CWE-614: Sensitive Cookie in HTTPS Session Without 'Secure' Attribute
* CWE-732: Incorrect Permission Assignment for Critical Resource
* CWE-776: Improper Restriction of Recursive Entity References in DTDs ('XML Bomb')
* CWE-942: Permissive Cross-domain Policy with Untrusted Domains
* CWE-1004: Sensitive Cookie Without 'HttpOnly' Flag
* CWE-1005: Sensitive Cookie Without 'SameSite' Attribute

### A03:2025 – Software Supply Chain Failures
* CWE-345: Insufficient Verification of Data Authenticity
* CWE-346: Origin Validation Error
* CWE-829: Inclusion of Functionality from Untrusted Control Sphere
* CWE-830: Inclusion of Web Functionality from an Untrusted Source
* CWE-937: OWASP Top Ten 2013 Category A9 - Using Components with Known Vulnerabilities
* CWE-1035: Single Sign-On (SSO) Support Issues
* CWE-1104: Use of Unmaintained Third Party Components
* CWE-1357: Reliance on Insufficiently Trustworthy Component

### A04:2025 – Cryptographic Failures
* CWE-259: Use of Hard-coded Password
* CWE-261: Weak Cryptography for Passwords
* CWE-295: Improper Certificate Validation
* CWE-310: Cryptographic Issues
* CWE-311: Missing Encryption of Sensitive Data
* CWE-312: Cleartext Storage of Sensitive Information
* CWE-313: Cleartext Storage in a File or on Disk
* CWE-314: Cleartext Storage in the Registry
* CWE-315: Cleartext Storage of Sensitive Information in a Cookie
* CWE-316: Cleartext Storage of Sensitive Information in Memory
* CWE-317: Cleartext Storage of Sensitive Information in GUI
* CWE-318: Cleartext Storage of Sensitive Information in Executable
* CWE-319: Cleartext Transmission of Sensitive Information
* CWE-321: Use of Hard-coded Cryptographic Key
* CWE-322: Key Exchange without Entity Authentication
* CWE-323: Reusing a Nonce, Key Pair in Negotiation
* CWE-324: Use of a Key Past its Expiration Date
* CWE-325: Missing Cryptographic Step
* CWE-326: Inadequate Encryption Strength
* CWE-327: Use of a Broken or Risky Cryptographic Algorithm
* CWE-328: Reversible One-Way Hash
* CWE-329: Not Using a Random IV with CBC Mode
* CWE-330: Use of Insufficiently Random Values
* CWE-331: Insufficient Entropy
* CWE-335: Incorrect Usage of Seeds in Pseudo-Random Number Generator
* CWE-336: Same Seed in Pseudo-Random Number Generator
* CWE-337: Predictable Seed in Pseudo-Random Number Generator
* CWE-338: Use of Cryptographically Weak Pseudo-Random Number Generator
* CWE-340: Predictability Problems
* CWE-347: Improper Verification of Cryptographic Signature
* CWE-757: Selection of Less-Secure Algorithm During Negotiation ('Algorithm Downgrade')
* CWE-759: Use of a One-Way Hash without a Salt
* CWE-760: Use of a One-Way Hash with a Predictable Salt
* CWE-798: Use of Hard-coded Credentials
* CWE-818: Insufficient Transport Layer Protection

### A05:2025 – Injection
* CWE-74: Improper Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')
* CWE-75: Failure to Sanitize Special Elements into a Different Plane (Special Element Injection)
* CWE-77: Improper Neutralization of Special Elements used in a Command ('Command Injection')
* CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')
* CWE-79: Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
* CWE-80: Improper Neutralization of Script-Related HTML Tags in a Web Page
* CWE-83: Improper Neutralization of Script in Attributes in a Web Page
* CWE-87: Improper Neutralization of Alternate XSS Syntax
* CWE-88: Improper Neutralization of Argument Delimiters in a Command
* CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
* CWE-90: Improper Neutralization of Special Elements used in an LDAP Query ('LDAP Injection')
* CWE-91: XML Injection
* CWE-93: Improper Neutralization of CRLF Sequences ('CRLF Injection')
* CWE-94: Improper Control of Generation of Code ('Code Injection')
* CWE-95: Improper Neutralization of Directives in Dynamically Evaluated Code ('Eval Injection')
* CWE-96: Improper Neutralization of Directives in Statically Saved Code ('Static Code Injection')
* CWE-97: Improper Neutralization of Server-Side Includes (SSI) Within a Web Page
* CWE-98: Improper Control of Filename for Include/Require Statement in PHP Program
* CWE-99: Improper Control of Resource Identifiers ('Resource Injection')
* CWE-113: Improper Neutralization of CRLF Sequences in HTTP Headers ('HTTP Response Splitting')
* CWE-116: Improper Encoding or Escaping of Output
* CWE-138: Improper Neutralization of Special Elements
* CWE-243: Creation of chroot Jail Without Changing Working Directory
* CWE-250: Execution with Unnecessary Privileges
* CWE-259: Use of Hard-coded Password
* CWE-400: Uncontrolled Resource Consumption
* CWE-404: Improper Resource Shutdown or Release
* CWE-564: SQL Injection: Hibernate
* CWE-611: Improper Restriction of XML External Entity Reference
* CWE-643: Improper Neutralization of Data within XPath Expressions ('XPath Injection')
* CWE-644: Improper Neutralization of HTTP Headers for Scripting Syntax
* CWE-652: Improper Neutralization of Data within XQuery Expressions ('XQuery Injection')
* CWE-917: Improper Neutralization of Special Elements used in an Expression Language Statement ('Expression Language Injection')

### A06:2025 – Insecure Design
* CWE-73: External Control of File Name or Path
* CWE-183: Permissive List of Allowed Inputs
* CWE-209: Generation of Error Messages Containing Sensitive Information
* CWE-213: Intentional Information Exposure
* CWE-256: Unprotected Storage of Credentials
* CWE-257: Storing Passwords in a Recoverable Format
* CWE-266: Incorrect Privilege Assignment
* CWE-269: Improper Privilege Management
* CWE-280: Improper Handling of Insufficient Permissions or Privileges
* CWE-311: Missing Encryption of Sensitive Data
* CWE-316: Cleartext Storage of Sensitive Information in Memory
* CWE-419: Unprotected Primary Channel
* CWE-430: Deployment of Wrong Handler
* CWE-434: Unrestricted Upload of File with Dangerous Type
* CWE-444: Inconsistent Interpretation of HTTP Requests
* CWE-451: User Interface (UI) Misrepresentation of Critical Information
* CWE-501: Trust Boundary Violation
* CWE-522: Insufficiently Protected Credentials
* CWE-539: Information Exposure Through Persistent Cookies
* CWE-579: J2EE Bad Practices: Non-serializable Object Stored in Session
* CWE-598: Use of GET Request Method With Sensitive Query Strings
* CWE-602: Client-Side Enforcement of Server-Side Security
* CWE-653: Improper Isolation or Compartmentalization
* CWE-656: Reliance on Security Through Obscurity
* CWE-657: Violation of Secure Design Principles
* CWE-799: Improper Control of Interaction Frequency
* CWE-807: Reliance on Untrusted Inputs in a Security Decision
* CWE-840: Business Logic Errors
* CWE-841: Improper Enforcement of Behavioral Workflow
* CWE-843: Access of Resource Using Incompatible Type
* CWE-1021: Improper Restriction of Rendered UI Layers or Frames
* CWE-1173: Improper Use of Validation Framework

### A07:2025 – Authentication Failures
* CWE-255: Credentials Management Errors
* CWE-259: Use of Hard-coded Password
* CWE-287: Improper Authentication
* CWE-288: Authentication Bypass Using an Alternate Path or Channel
* CWE-290: Authentication Bypass by Spoofing
* CWE-294: Authentication Bypass by Capture-replay
* CWE-295: Improper Certificate Validation
* CWE-297: Improper Validation of Certificate with Host Mismatch
* CWE-300: Channel Accessible by Non-Endpoint
* CWE-302: Authentication Bypass by Assumed-Immutable Data
* CWE-303: Incorrect Implementation of Authentication Algorithm
* CWE-304: Missing Critical Step in Authentication
* CWE-305: Authentication Bypass by Primary Weakness
* CWE-306: Missing Authentication for Critical Function
* CWE-307: Improper Restriction of Excessive Authentication Attempts
* CWE-308: Use of Single-factor Authentication
* CWE-346: Origin Validation Error
* CWE-384: Session Fixation
* CWE-522: Insufficiently Protected Credentials
* CWE-613: Insufficient Session Expiration
* CWE-620: Unverified Password Change
* CWE-640: Weak Password Recovery Mechanism for Forgotten Password
* CWE-798: Use of Hard-coded Credentials
* CWE-940: Improper Verification of Source of a Communication Channel
* CWE-1216: Lockout Mechanism Errors

### A08:2025 – Software or Data Integrity Failures
* CWE-345: Insufficient Verification of Data Authenticity
* CWE-353: Missing Support for Integrity Check
* CWE-426: Untrusted Search Path
* CWE-494: Download of Code Without Integrity Check
* CWE-502: Deserialization of Untrusted Data
* CWE-565: Reliance on Cookies without Validation and Integrity Checking
* CWE-784: Reliance on Cookies without Validation and Integrity Checking in a Security Decision
* CWE-829: Inclusion of Functionality from Untrusted Control Sphere
* CWE-830: Inclusion of Web Functionality from an Untrusted Source
* CWE-915: Improperly Controlled Modification of Dynamically-Determined Object Attributes

### A09:2025 – Security Logging and Alerting Failures
* CWE-117: Improper Output Neutralization for Logs
* CWE-223: Omission of Security-relevant Information
* CWE-532: Insertion of Sensitive Information into Log File
* CWE-778: Insufficient Logging
* CWE-779: Logging of Excessive Data

### A10:2025 – Mishandling of Exceptional Conditions
* CWE-209: Generation of Error Messages Containing Sensitive Information
* CWE-248: Uncaught Exception
* CWE-273: Improper Check for Dropped Privileges
* CWE-280: Improper Handling of Insufficient Permissions or Privileges
* CWE-390: Detection of Error Condition Without Action
* CWE-391: Unchecked Error Condition
* CWE-392: Missing Report of Error Condition
* CWE-393: Return of Wrong Status Code
* CWE-394: Unexpected Status Code or Return Value
* CWE-395: Use of NullPointerException Catch to Detect NULL Pointer Dereference
* CWE-396: Declaration of Catch for Generic Exception
* CWE-397: Declaration of Throws for Generic Exception
* CWE-430: Deployment of Wrong Handler
* CWE-433: Unparsed Raw Web Content Delivery
* CWE-544: Missing Standardized Error Handling Mechanism
* CWE-600: Uncaught Exception in Servlet
* CWE-601: URL Redirection to Untrusted Site ('Open Redirect')
* CWE-754: Improper Check for Unusual or Exceptional Conditions
* CWE-755: Improper Handling of Exceptional Conditions

---

## 2. OWASP Top 10:2021

### A01:2021 – Broken Access Control
*(Uses the exact same set of CWEs mapped in A01:2025, excluding CWE-918 SSRF which was its own category in 2021).*
* CWE-22, CWE-23, CWE-24, CWE-25, CWE-26, CWE-27, CWE-28, CWE-29, CWE-30, CWE-31, CWE-32, CWE-33, CWE-34, CWE-35, CWE-55, CWE-56, CWE-57, CWE-58, CWE-59, CWE-60, CWE-200, CWE-219, CWE-264, CWE-275, CWE-276, CWE-284, CWE-285, CWE-352, CWE-359, CWE-434, CWE-552, CWE-639, CWE-732, CWE-862, CWE-863.

### A02:2021 – Cryptographic Failures
*(Matches A04:2025)*
* CWE-259, CWE-261, CWE-295, CWE-310, CWE-311, CWE-312, CWE-313, CWE-314, CWE-315, CWE-316, CWE-317, CWE-318, CWE-319, CWE-321, CWE-322, CWE-323, CWE-324, CWE-325, CWE-326, CWE-327, CWE-328, CWE-329, CWE-330, CWE-331, CWE-335, CWE-336, CWE-337, CWE-338, CWE-340, CWE-347, CWE-757, CWE-759, CWE-760, CWE-798, CWE-818.

### A03:2021 – Injection
*(Matches A05:2025)*
* CWE-74, CWE-75, CWE-77, CWE-78, CWE-79, CWE-80, CWE-83, CWE-87, CWE-88, CWE-89, CWE-90, CWE-91, CWE-93, CWE-94, CWE-95, CWE-96, CWE-97, CWE-98, CWE-99, CWE-113, CWE-116, CWE-138, CWE-243, CWE-250, CWE-259, CWE-400, CWE-404, CWE-564, CWE-611, CWE-643, CWE-644, CWE-652, CWE-917.

### A04:2021 – Insecure Design
*(Matches A06:2025)*
* CWE-73, CWE-183, CWE-209, CWE-213, CWE-256, CWE-257, CWE-266, CWE-269, CWE-280, CWE-311, CWE-316, CWE-419, CWE-430, CWE-434, CWE-444, CWE-451, CWE-501, CWE-522, CWE-539, CWE-579, CWE-598, CWE-602, CWE-653, CWE-656, CWE-657, CWE-799, CWE-807, CWE-840, CWE-841, CWE-843, CWE-1021, CWE-1173.

### A05:2021 – Security Misconfiguration
*(Matches A02:2025)*
* CWE-16, CWE-209, CWE-215, CWE-237, CWE-383, CWE-536, CWE-548, CWE-611, CWE-614, CWE-732, CWE-776, CWE-942, CWE-1004, CWE-1005.

### A06:2021 – Vulnerable and Outdated Components
* CWE-937: OWASP Top Ten 2013 Category A9 - Using Components with Known Vulnerabilities
* CWE-1035: Single Sign-On (SSO) Support Issues
* CWE-1104: Use of Unmaintained Third Party Components

### A07:2021 – Identification and Authentication Failures
*(Matches A07:2025)*
* CWE-255, CWE-259, CWE-287, CWE-288, CWE-290, CWE-294, CWE-295, CWE-297, CWE-300, CWE-302, CWE-303, CWE-304, CWE-305, CWE-306, CWE-307, CWE-308, CWE-346, CWE-384, CWE-522, CWE-613, CWE-620, CWE-640, CWE-798, CWE-940, CWE-1216.

### A08:2021 – Software and Data Integrity Failures
*(Matches A08:2025)*
* CWE-345, CWE-353, CWE-426, CWE-494, CWE-502, CWE-565, CWE-784, CWE-829, CWE-830, CWE-915.

### A09:2021 – Security Logging and Monitoring Failures
*(Matches A09:2025)*
* CWE-117, CWE-223, CWE-532, CWE-778, CWE-779.

### A10:2021 – Server-Side Request Forgery (SSRF)
* CWE-918: Server-Side Request Forgery (SSRF)

---

## 3. OWASP Top 10:2017
*Note: The 2017 mappings were less exhaustive and relied on targeted primary CWEs.*

### A1:2017 – Injection
* CWE-77: Command Injection
* CWE-89: SQL Injection
* CWE-564: SQL Injection: Hibernate
* CWE-917: Expression Language Injection

### A2:2017 – Broken Authentication
* CWE-287: Improper Authentication
* CWE-384: Session Fixation

### A3:2017 – Sensitive Data Exposure
* CWE-319: Cleartext Transmission of Sensitive Information
* CWE-327: Broken or Risky Cryptographic Algorithm
* CWE-200: Exposure of Sensitive Information

### A4:2017 – XML External Entities (XXE)
* CWE-611: Improper Restriction of XML External Entity Reference
* CWE-827: Improper Control of Document Type Definition

### A5:2017 – Broken Access Control
* CWE-22: Path Traversal
* CWE-284: Improper Access Control
* CWE-285: Improper Authorization
* CWE-639: Authorization Bypass Through User-Controlled Key

### A6:2017 – Security Misconfiguration
* CWE-16: Configuration
* CWE-209: Generation of Error Messages Containing Sensitive Information
* CWE-732: Incorrect Permission Assignment for Critical Resource

### A7:2017 – Cross-Site Scripting (XSS)
* CWE-79: Cross-site Scripting (XSS)

### A8:2017 – Insecure Deserialization
* CWE-502: Deserialization of Untrusted Data

### A9:2017 – Using Components with Known Vulnerabilities
* CWE-1035: Single Sign-On Support Issues
* CWE-937: Using Components with Known Vulnerabilities

### A10:2017 – Insufficient Logging & Monitoring
* CWE-778: Insufficient Logging
* CWE-223: Omission of Security-relevant Information