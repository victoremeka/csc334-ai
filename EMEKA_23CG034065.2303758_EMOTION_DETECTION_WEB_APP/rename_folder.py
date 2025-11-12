"""
Folder Rename Helper Script
This script helps rename the project folder to match student details
Format: SURNAME_MAT.XXXXXX
"""

import os
import sys
import shutil
from pathlib import Path

def print_header():
    """Print header"""
    print("\n" + "=" * 70)
    print("  FOLDER RENAME HELPER - Emotion Detection Project")
    print("=" * 70 + "\n")

def get_current_folder():
    """Get the current folder name"""
    current_path = Path(__file__).parent.absolute()
    folder_name = current_path.name
    return current_path, folder_name

def validate_surname(surname):
    """Validate surname input"""
    if not surname:
        return False, "Surname cannot be empty"
    if not surname.replace('_', '').isalpha():
        return False, "Surname should contain only letters"
    return True, ""

def validate_matric(matric):
    """Validate matriculation number"""
    if not matric:
        return False, "Matriculation number cannot be empty"
    # Remove any dots or slashes
    matric_clean = matric.replace('.', '').replace('/', '').replace('-', '')
    if len(matric_clean) < 6:
        return False, "Matriculation number seems too short"
    return True, ""

def create_new_folder_name(surname, matric):
    """Create new folder name in correct format"""
    # Clean inputs
    surname_clean = surname.upper().strip().replace(' ', '_')
    matric_clean = matric.strip()
    
    # Ensure matric has MAT prefix if not already
    if not matric_clean.upper().startswith('MAT'):
        matric_clean = f"MAT.{matric_clean}"
    elif 'MAT' in matric_clean.upper() and '.' not in matric_clean:
        matric_clean = matric_clean.upper().replace('MAT', 'MAT.')
    
    new_name = f"{surname_clean}_{matric_clean}"
    return new_name

def rename_folder(old_path, new_name):
    """Rename the folder"""
    try:
        parent_dir = old_path.parent
        new_path = parent_dir / new_name
        
        # Check if target already exists
        if new_path.exists():
            print(f"\n⚠ Warning: Folder '{new_name}' already exists!")
            choice = input("Do you want to overwrite it? (yes/no): ").strip().lower()
            if choice != 'yes':
                print("Rename cancelled.")
                return False
            shutil.rmtree(new_path)
        
        # Rename the folder
        os.rename(old_path, new_path)
        print(f"\n✓ Folder successfully renamed!")
        print(f"  Old name: {old_path.name}")
        print(f"  New name: {new_path.name}")
        print(f"\n  New path: {new_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error renaming folder: {e}")
        return False

def main():
    """Main function"""
    print_header()
    
    current_path, current_name = get_current_folder()
    
    print(f"Current folder name: {current_name}")
    print(f"Current path: {current_path}\n")
    
    if current_name != "STUDENT_MAT.12345678":
        print("Note: This folder has already been renamed.")
        print("Current name:", current_name)
        choice = input("\nDo you want to rename it again? (yes/no): ").strip().lower()
        if choice != 'yes':
            print("\nExiting...")
            return
    
    print("\nPlease provide your details:")
    print("-" * 70)
    
    # Get surname
    while True:
        surname = input("\nEnter your SURNAME (all caps): ").strip()
        valid, message = validate_surname(surname)
        if valid:
            break
        else:
            print(f"✗ {message}. Please try again.")
    
    # Get matriculation number
    while True:
        print("\nEnter your Matriculation Number")
        print("(Examples: 23CG034065, MAT.12345678, 12345678)")
        matric = input("Matric Number: ").strip()
        valid, message = validate_matric(matric)
        if valid:
            break
        else:
            print(f"✗ {message}. Please try again.")
    
    # Create new folder name
    new_name = create_new_folder_name(surname, matric)
    
    # Confirm
    print("\n" + "=" * 70)
    print("CONFIRMATION")
    print("=" * 70)
    print(f"\nCurrent folder: {current_name}")
    print(f"New folder:     {new_name}")
    print(f"\nFull new path:  {current_path.parent / new_name}")
    
    choice = input("\n\nIs this correct? (yes/no): ").strip().lower()
    
    if choice == 'yes':
        success = rename_folder(current_path, new_name)
        if success:
            print("\n" + "=" * 70)
            print("SUCCESS!")
            print("=" * 70)
            print("\nYour project folder has been renamed successfully.")
            print("\nIMPORTANT NOTES:")
            print("1. You may need to close and reopen your terminal/IDE")
            print("2. Update any bookmarks or shortcuts to the new path")
            print("3. If using Git, commit the renamed folder")
            print("\nYour project is now ready for submission! 🎉")
        else:
            print("\nRename failed. Please try again or rename manually.")
    else:
        print("\nRename cancelled. No changes made.")
    
    print("\n" + "=" * 70)
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)