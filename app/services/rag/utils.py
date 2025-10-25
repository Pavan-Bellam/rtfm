import asyncio
from pathlib import Path
from typing import List, Optional
from tempfile import NamedTemporaryFile
import shutil

from app.core.logging import get_logger
from app.core.settings import settings

logger = get_logger(__name__)

def number_file(file_path: Path) -> None:
    """
    Add line numbers to lines in the file.
    Args:
        file_path : Path of the file that needs to be modified
    Returns:
        None
    Exceptions:
        FileNotFoundError, PermissionError,  UnicodeError, shutil.Error, OSError
    """
    try:
        with open(file_path, "r", encoding="utf-8") as src, NamedTemporaryFile("w", delete=False, encoding='utf-8') as tmp:
            chars_count = 0
            code_block = False
            code_block_start_line = None
            
            for i, line in enumerate(src, 1):
                new_line = f"#{i}: {line}"
                
                # Check for code block BEFORE deciding to split
                if "```" in line:
                    code_block = not code_block
                
                # Check if we should split BEFORE writing this line
                estimated_tokens = chars_count // 4
                if estimated_tokens > settings.SPLIT_SIZE and not code_block:
                    tmp.write("<Chunk_Break>\n")
                    chars_count = 0
                
                # Now write the line
                tmp.write(new_line)
                chars_count += len(new_line)
            
            # Warn if code block never closed
            if code_block:
                logger.warning(
                    "Code block never closed",
                    extra={
                        "extra_fields":{
                            "file_path": file_path
                        }
                    }
                )

        shutil.move(tmp.name, file_path)

    except FileNotFoundError as e:
        logger.error(
            "file not found",
            exc_info=True,
            extra={
                "extra_fields" : {
                    "file_path" : file_path
                }
            }
        )
        raise

    except PermissionError as e:
        logger.error(
            "Permission denied for the file",
            exc_info=True,
            extra={
                "extra_fields" : {
                    "file_path" : file_path
                }
            }
        )
        raise

    except UnicodeError as e:
        logger.error(
            "Encoding issue",
            exc_info=True,
            extra={
                "extra_fields": {
                    "file_path" : file_path
                }
            }
        )
        raise
    except shutil.Error as e:
        logger.error(
            "I/O issue with shuttule",
            exc_info=True,
            extra={
                "extra_fields": {
                    "file_path" : file_path
                }
            }
        )
        raise
    except OSError as e:
        logger.error(
            "I/O issue",
            exc_info=True,
            extra={
                "extra_fields": {
                    "file_path" : file_path
                }
            }
        )
        raise

    finally:
        if tmp and Path(tmp.name).exists():
            try:
                Path(tmp.name).unlink()
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to remove temp file",
                    exc_info=True,
                    extra={
                        "extra_fields" : {
                            "temp_file_name": tmp.name
                        }
                    }
                )


async def number_files(files_list: Optional[List[Path]] = None, files_dir: Optional[str] = None) -> None:
    """
    Add line numbers to multiple files either from a provided list or by discovering files in a directory.
    
    Args:
        files_list: Optional list of file paths to process
        files_dir: Optional directory path to discover and process files from
        
    Returns:
        None
        
    Raises:
        ValueError: If both parameters are None or both are provided
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If access to directory is denied
        OSError: For other file system related errors
    """
    try:
        # Validate input parameters
        if files_list is None and files_dir is None:
            logger.error("Both files_list and files_dir are None")
            raise ValueError("Either files_list or files_dir must be provided, not both None")
        
        if files_list is not None and files_dir is not None:
            logger.error("Both files_list and files_dir are provided")
            raise ValueError("Provide either files_list or files_dir, not both")
        
        # Determine which files to process
        files_to_process: List[Path] = []
        
        if files_list is not None:
            # Validate that all files in the list exist
            for file_path in files_list:
                if not file_path.exists():
                    logger.error(
                        "File not found",
                        exc_info=True,
                        extra={
                            "extra_fields": {
                                "file_path": file_path
                            }
                        }
                    )
                    raise FileNotFoundError(f"File not found: {file_path}")
                if not file_path.is_file():
                    logger.error(
                        "Path is not a file",
                        exc_info=True,
                        extra={
                            "extra_fields": {
                                "file_path": file_path
                            }
                        }
                    )
                    raise ValueError(f"Path is not a file: {file_path}")
            files_to_process = files_list
            
        elif files_dir is not None:
            # Validate directory exists and is accessible
            dir_path = Path(files_dir)
            if not dir_path.exists():
                logger.error(
                    "Directory not found",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "directory_path": files_dir
                        }
                    }
                )
                raise FileNotFoundError(f"Directory not found: {files_dir}")
            
            if not dir_path.is_dir():
                logger.error(
                    "Path is not a directory",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "directory_path": files_dir
                        }
                    }
                )
                raise ValueError(f"Path is not a directory: {files_dir}")
            
            # Discover files in the directory (recursively)
            try:
                files_to_process = [f for f in dir_path.rglob("*") if f.is_file()]
                if not files_to_process:
                    logger.warning(
                        "No files found in directory",
                        extra={
                            "extra_fields": {
                                "directory_path": files_dir
                            }
                        }
                    )
                    return
            except PermissionError as e:
                logger.error(
                    "Permission denied accessing directory",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "directory_path": files_dir
                        }
                    }
                )
                raise
            except OSError as e:
                logger.error(
                    "OS error accessing directory",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "directory_path": files_dir
                        }
                    }
                )
                raise
        
        # Process files asynchronously
        if not files_to_process:
            logger.warning(
                "No files to process",
                extra={
                    "extra_fields": {
                        "files_list": str(files_list) if files_list else None,
                        "files_dir": files_dir
                    }
                }
            )
            return
            
        logger.info(
            "Processing files",
            extra={
                "extra_fields": {
                    "file_count": len(files_to_process)
                }
            }
        )
        
        # Create tasks for processing files
        number_files_tasks = [
            asyncio.to_thread(number_file, file_path) 
            for file_path in files_to_process
        ]
        
        # Execute all tasks and collect results
        results = await asyncio.gather(*number_files_tasks, return_exceptions=True)
        
        # Check for any exceptions in the results
        failed_files = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_files.append((files_to_process[i], result))
                logger.error(
                    "Failed to process file",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "file_path": str(files_to_process[i]),
                            "error": str(result)
                        }
                    }
                )
        
        if failed_files:
            logger.warning(
                "Some files failed to process",
                extra={
                    "extra_fields": {
                        "failed_count": len(failed_files),
                        "total_count": len(files_to_process)
                    }
                }
            )
        else:
            logger.info(
                "Successfully processed all files",
                extra={
                    "extra_fields": {
                        "processed_count": len(files_to_process)
                    }
                }
            )
            
    except ValueError as e:
        logger.error(
            "Invalid input parameters",
            exc_info=True,
            extra={
                "extra_fields": {
                    "error": str(e)
                }
            }
        )
        raise
    except FileNotFoundError as e:
        logger.error(
            "File or directory not found",
            exc_info=True,
            extra={
                "extra_fields": {
                    "error": str(e)
                }
            }
        )
        raise
    except PermissionError as e:
        logger.error(
            "Permission denied",
            exc_info=True,
            extra={
                "extra_fields": {
                    "error": str(e)
                }
            }
        )
        raise
    except OSError as e:
        logger.error(
            "OS error",
            exc_info=True,
            extra={
                "extra_fields": {
                    "error": str(e)
                }
            }
        )
        raise
    except Exception as e:
        logger.error(
            "Unexpected error while numbering files",
            exc_info=True,
            extra={
                "extra_fields": {
                    "files_list": str(files_list) if files_list else None,
                    "files_dir": files_dir
                }
            }
        )
        raise



