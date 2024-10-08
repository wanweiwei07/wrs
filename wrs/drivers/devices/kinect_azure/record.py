from wrs import drivers as _k4a_record


class Record(object):

    def __init__(self, lib_path, device_handle, device_configuration, filepath):
        self.k4arecord = _k4a_record.k4arecord(lib_path)
        self.record_handle = _k4a_record.k4a_record_t()
        self.header_written = False
        self.create_recording(device_handle, device_configuration, filepath)

    def __del__(self):
        self.close()

    def create_recording(self, device_handle, device_configuration, filepath):
        _k4a_record.VERIFY(
            self.k4arecord.k4a_record_create(filepath.encode('utf-8'), device_handle, device_configuration,
                                             self.record_handle), "Failed to create recording!")

    def is_valid(self):
        return self.record_handle != None

    def close(self):
        if self.is_valid():
            self.k4arecord.k4a_record_close(self.record_handle)
            self.record_handle = None

    def flush(self):
        if self.is_valid():
            _k4a_record.VERIFY(self.k4arecord.k4a_record_flush(self.record_handle), "Failed to flush!")

    def write_header(self):
        if self.is_valid():
            _k4a_record.VERIFY(self.k4arecord.k4a_record_write_header(self.record_handle), "Failed to write header!")

    def write_capture(self, capture_handle):
        if not self.is_valid():
            raise NameError('Recording not found')
        if not self.header_written:
            self.write_header()
            self.header_written = True
        _k4a_record.VERIFY(self.k4arecord.k4a_record_write_capture(self.record_handle, capture_handle),
                           "Failed to write capture!")
