import os
import sys


from PSZS.Utils.io.filewriter import get_outdir

class TextLogger(object):
    """Writes stream output to external text file.

    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self, keepalive: bool = False):
        if keepalive is False:
            self.terminal.close()
        self.log.close()

class Logger:
    """
    A useful logger that

    - writes outputs to files and displays them on the console at the same time.
    - manages the directory of checkpoints and debugging images.

    Args:
        root (str): the root directory of logger

    """

    def __init__(self, root: str, exp_name: str, 
                 sep_chk: bool=False, log_filename: str = "log.txt",
                 create_checkpoint: bool = True,):
        if log_filename[-4:] != '.txt':
            log_filename += '.txt'
        self.out_dir = get_outdir(root, exp_name, use_uuid=True)
        if sep_chk:
            self.checkpoint_directory = get_outdir(root, 'checkpoints', exp_name)
        else:
            self.checkpoint_directory = os.path.join(self.out_dir, 'checkpoints')
        self.epoch = 0
        self.visualization_dir = os.path.join(self.out_dir, 'visualizations')

        os.makedirs(self.out_dir, exist_ok=True)
        if create_checkpoint:
            os.makedirs(self.checkpoint_directory, exist_ok=True)

        # redirect std out
        log_filename = os.path.join(self.out_dir, log_filename)
        # Remove file extension (guaranteed to be .txt)
        unique_filepath = log_filename
        for i in range(100):
            if os.path.exists(unique_filepath):
                unique_filepath = log_filename[:-4] + f'_{i}' + log_filename[-4:]
        log_filename = unique_filepath
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger

    def set_epoch(self, epoch: int):
        """Set the epoch number."""
        # os.makedirs(os.path.join(self.visualize_directory, str(epoch)), exist_ok=True)
        self.epoch = epoch
        
    def get_visualization_dir(self) -> str:
        os.makedirs(self.visualization_dir, exist_ok=True)
        return self.visualization_dir

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.

        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, checkpoint will be saved to ``Epoch_{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.

        """
        if name is None:
            name = f'Epoch_{self.epoch}'
        name = str(name)
        if name[-4:] != '.pth':
            name += '.pth'
        return os.path.join(self.checkpoint_directory, name)

    def close(self, keepalive: bool = False):
        sys.stdout =  self.original_stdout
        sys.stderr = self.original_stderr
        self.logger.close(keepalive=keepalive)
