using HDF5

"""
Write an ODIL state to an HDF5 file, including the stored solution vector and the last
included time index.
"""
function write_h5(state::OdilState, filename::String)
    h5open(filename, "w") do file
        HDF5.attributes(file)["it_last"] = state.it_last
        file["u"] = state.u
    end
end

"""
Read an ODIL state from an HDF5 file and reconstruct the corresponding `OdilState`.
"""
function read_h5(filename::String)
    h5open(filename, "r") do file
        it_last = HDF5.read(HDF5.attributes(file)["it_last"])
        u = HDF5.read(file["u"])
        return OdilState(u, it_last)
    end
end