use arrow::{
    file::{writer::FileWriter, write_all, Writer},
    record_batch::RecordBatch,
    util::hash::XXHash64,
};
use std::fs::File;

fn hash_text_column(input_path: &str, output_path: &str) {
    let mut input_reader = FileReader::try_new(input_path).unwrap();
    let input_schema = input_reader.schema().clone();
    let output_file = File::create(output_path).unwrap();
    let mut output_writer = FileWriter::try_new(input_schema, output_file).unwrap();

    let hash_builder = XXHash64::builder().seed(0);

    while let Some(result) = input_reader.next() {
        let batch = result.unwrap();

        // Get the "text" column
        let text_column = batch
            .column(batch.schema().field_with_name("text").unwrap().index())
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        // Create a new UInt64Array for the hash values
        let hash_values = arrow::array::UInt64Array::new(text_column.len(), text_column.data().len());

        // Hash the text values and write the result to the new array
        for i in 0..text_column.len() {
            let text = text_column.value(i);
            let hash = hash_builder.hash(text.as_bytes());
            hash_values.set(i, hash);
        }

        // Add the new column to the batch's schema
        let new_field = arrow::field("hash", arrow::DataType::UInt64, true);
        let new_schema = batch
            .schema()
            .add_field(new_field)
            .clone();

        // Create a new record batch with the new column
        let new_batch = RecordBatch::new(
            new_schema,
            vec![batch.column(0).clone(), hash_values.into()],
        );

        // Write the new batch to the output file
        write_all(&mut output_writer, vec![&new_batch]).unwrap();
    }
}
