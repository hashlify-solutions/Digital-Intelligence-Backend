import zipfile
import os
import xml.etree.ElementTree as ET
import json


class UfdrExtracter:
    """
    A class to extract and parse UFDR (Universal Forensic Data Reader) files.
    This class handles extraction of structured data and media files from UFDR archives,
    which are commonly used in digital forensics investigations.
    """
    
    # Data types to extract
    DATA_TYPES = [
        "Call",
        "Chat",
        "Email",
        "Location",
        "MMS",
        "Note",
        "SearchedItem",
        "SMS",
        "UserAccount",
    ]

    MEDIA_TYPES = {
        "photo": [
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
            ".heic", ".heif", ".raw", ".dng", ".cr2", ".nef", ".arw", ".orf",
            ".rw2", ".pef", ".sr2", ".raf", ".3fr", ".fff", ".dcr", ".kdc",
            ".srf", ".mrw", ".nrw", ".rwl", ".iiq", ".x3f", ".gpr",
        ],
        "video": [
            ".mp4", ".mov", ".3gp", ".avi", ".webm", ".mkv", ".flv", ".wmv",
            ".asf", ".m4v", ".3g2", ".ts", ".mts", ".m2ts", ".vob", ".divx",
            ".xvid", ".rm", ".rmvb", ".ogv", ".f4v",
        ],
        "audio": [
            ".amr", ".mp3", ".wav", ".m4a", ".aac", ".wma", ".ogg", ".3ga",
            ".awb", ".flac", ".opus", ".gsm", ".qcp", ".evrc", ".amr-wb",
            ".amr-nb", ".evs", ".silk", ".speex", ".vorbis", ".ac3", ".eac3",
            ".dts", ".pcm", ".aiff", ".au", ".snd", ".caf", ".adts", ".mp2",
            ".mpa", ".ra", ".wv", ".tta", ".ape", ".mka",
        ],
    }

    def __init__(self):
        """Initialize the UfdrExtracter with media extensions list."""
        self.all_media_extensions = [ext for exts in self.MEDIA_TYPES.values() for ext in exts]

    def extract_report_xml(self, ufdr_file_path, output_dir_path):
        """
        Extract the report.xml file from the UFDR archive.
        
        Args:
            ufdr_file_path (str): Path to the UFDR file
            
        Returns:
            str: Path to the extracted report.xml file
            
        Raises:
            FileNotFoundError: If report.xml is not found in the archive
        """
        with zipfile.ZipFile(ufdr_file_path, "r") as z:
            for name in z.namelist():
                if name.endswith("report.xml"):
                    extracted_report_xml = z.extract(name, output_dir_path)
                    return extracted_report_xml
        raise FileNotFoundError("report.xml not found in UFDR file")

    def get_namespace(self, element):
        """
        Extract namespace from XML element tag.
        
        Args:
            element: XML element
            
        Returns:
            str: Namespace URI or empty string if no namespace
        """
        if element.tag[0] == "{":
            return element.tag[1:].split("}")[0]
        return ""

    def extract_fields(self, elem, ns):
        """
        Extract fields from XML element recursively.
        
        Args:
            elem: XML element to extract fields from
            ns: Namespace dictionary
            
        Returns:
            dict: Extracted field data
        """
        data = {}
        
        # Simple fields
        for field in elem.findall("ns:field", ns) if ns else elem.findall("field"):
            fname = field.attrib.get("name")
            value_elem = field.find("ns:value", ns) if ns else field.find("value")
            if value_elem is not None and value_elem.text is not None:
                data[fname] = value_elem.text.strip()
            else:
                empty_elem = field.find("ns:empty", ns) if ns else field.find("empty")
                data[fname] = None if empty_elem is not None else ""
                
        # modelField (single nested model)
        for mfield in (
            elem.findall("ns:modelField", ns) if ns else elem.findall("modelField")
        ):
            fname = mfield.attrib.get("name")
            model = mfield.find("ns:model", ns) if ns else mfield.find("model")
            if model is not None:
                data[fname] = self.extract_fields(model, ns) if ns else self.extract_fields(model, {})
            else:
                data[fname] = None
                
        # multiModelField (list of nested models)
        for mmfield in (
            elem.findall("ns:multiModelField", ns)
            if ns
            else elem.findall("multiModelField")
        ):
            fname = mmfield.attrib.get("name")
            models = []
            for model in (
                mmfield.findall("ns:model", ns) if ns else mmfield.findall("model")
            ):
                models.append(
                    self.extract_fields(model, ns) if ns else self.extract_fields(model, {})
                )
            data[fname] = models
            
        return data

    def parse_report_xml(self, report_xml_path):
        """
        Parse the report.xml file and extract structured data.
        
        Args:
            report_xml_path (str): Path to the report.xml file
            
        Returns:
            dict: Parsed data organized by data type
        """
        tree = ET.parse(report_xml_path)
        root = tree.getroot()
        ns_uri = self.get_namespace(root)
        ns = {"ns": ns_uri} if ns_uri else {}
        decoded_data = root.find("ns:decodedData", ns) if ns else root.find("decodedData")
        results = {}
        
        if decoded_data is None:
            print(
                f"Warning: <decodedData> section not found in report.xml. Namespace used: {ns_uri}"
            )
            return results
            
        for model_type in (
            decoded_data.findall("ns:modelType", ns)
            if ns
            else decoded_data.findall("modelType")
        ):
            dtype = model_type.attrib.get("type")
            if dtype in self.DATA_TYPES:
                items = []
                for model in (
                    model_type.findall("ns:model", ns)
                    if ns
                    else model_type.findall("model")
                ):
                    item = self.extract_fields(model, ns) if ns else self.extract_fields(model, {})
                    items.append(item)
                results[dtype] = items
                
        return results

    def save_json(self, data, out_dir):
        """
        Save extracted data to JSON files.
        
        Args:
            data (dict): Data to save, organized by type
            out_dir (str): Output directory
        """
        os.makedirs(out_dir, exist_ok=True)
        for dtype, items in data.items():
            out_path = os.path.join(out_dir, f"{dtype}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"Saved {dtype} data to {out_path}")

    def get_media_type(self, ext):
        """
        Determine media type based on file extension.
        
        Args:
            ext (str): File extension
            
        Returns:
            str: Media type ('photo', 'video', 'audio', or 'other')
        """
        for mtype, exts in self.MEDIA_TYPES.items():
            if ext in exts:
                return mtype
        return "other"

    def extract_media_files(self, ufdr_path, out_dir):
        """
        Extract media files from UFDR archive.
        
        Args:
            ufdr_path (str): Path to the UFDR file
            out_dir (str): Output directory for media files
        """
        media_dir = os.path.join(out_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        
        with zipfile.ZipFile(ufdr_path, "r") as z:
            media_files = [
                name
                for name in z.namelist()
                if os.path.splitext(name)[1].lower() in self.all_media_extensions
            ]
            
            for name in media_files:
                ext = os.path.splitext(name)[1].lower()
                mtype = self.get_media_type(ext)
                type_dir = os.path.join(media_dir, mtype)
                os.makedirs(type_dir, exist_ok=True)
                base_name = os.path.basename(name)
                
                # Skip macOS resource fork files (._prefix) and hidden files
                if base_name.startswith("._") or base_name.startswith("."):
                    continue
                
                out_path = os.path.join(type_dir, base_name)
                
                with z.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                print(f"Extracted {mtype}: {out_path}")
                
        if not media_files:
            print("No media files found in UFDR archive.")

    def extract_ufdr(self, ufdr_file_path, output_dir_path):
        """
        Main method to parse a UFDR file and extract all data.
        
        Args:
            ufdr_file_path (str): Path to the UFDR file
            output_dir_path (str): Output directory for extracted data
            
        Returns:
            dict: Extracted structured data
        """
        print(f"Extracting report.xml from {ufdr_file_path}...")
        report_xml = self.extract_report_xml(ufdr_file_path, output_dir_path)
        
        print(f"Parsing {report_xml}...")
        data = self.parse_report_xml(report_xml)
        
        print(f"Saving JSON output to {output_dir_path}...")
        self.save_json(data, output_dir_path)
        
        print("Extracting media files...")
        self.extract_media_files(ufdr_file_path, output_dir_path)
        
    