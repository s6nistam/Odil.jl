using HDF5

function write_h5(state::OdilState, filename::String)
    h5open(filename, "w") do file
        HDF5.attributes(file)["it_last"] = state.it_last
        file["u"] = state.u
    end
end

function read_h5(filename::String)
    h5open(filename, "r") do file
        it_last = HDF5.read(HDF5.attributes(file)["it_last"])
        u = HDF5.read(file["u"])
        return OdilState(u, it_last)
    end
end