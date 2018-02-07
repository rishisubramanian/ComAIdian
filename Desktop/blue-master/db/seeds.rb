# This file should contain all the record creation needed to seed the database with its default values.
# The data can then be loaded with the rails db:seed command (or created alongside the database with db:setup).
#
# Examples:
#
#   movies = Movie.create([{ name: 'Star Wars' }, { name: 'Lord of the Rings' }])
#   Character.create(name: 'Luke', movie: movies.first)

connection = ActiveRecord::Base.connection
sql = File.read('db/jokedb.sql')
statements = sql.split(/$/)
statements.pop
statements.pop
statements.shift


ActiveRecord::Base.transaction do
  statements.each do |statement|
    statement.chomp!("\n")
    statement.sub!(/^\n/, '')
    # statement.chomp!(';')
    connection.execute(statement)
  end
end